"""
test_ssurgo_api.py
------------------
Standalone diagnostic for the USDA SSURGO Soil Data Access API.
Tests connectivity, endpoint responsiveness, and a real Iowa field query
independent of the CoverMap app.

Run:
    python test_ssurgo_api.py

No field boundary upload required — uses a hardcoded Shelby County, Iowa
polygon (near Harlan) as a known-good test case.
"""

import sys
import json
import time
import requests

API_URL = "https://SDMDataAccess.nrcs.usda.gov/Tabular/post.rest"
TIMEOUT = 20

# Small test polygon — Shelby County Iowa, near Harlan (EPSG:4326, lon lat)
# Roughly 40 acres of cropland on Monona silt loam
TEST_WKT = (
    "POLYGON(("
    "-95.3350 41.6520,"
    "-95.3300 41.6520,"
    "-95.3300 41.6490,"
    "-95.3350 41.6490,"
    "-95.3350 41.6520"
    "))"
)

# Minimal ping query — returns nothing real, just verifies the endpoint is alive
PING_QUERY = "SELECT TOP 1 mukey FROM mapunit"

# Real K-factor query matching wss_utils.py
KFACTOR_QUERY = f"""SELECT TOP 1
    mu.muname, c.compname, c.comppct_r,
    (SELECT TOP 1 kwfact FROM chorizon ch
     JOIN component c2 ON ch.cokey = c2.cokey
     WHERE c2.cokey = c.cokey
     AND ch.hzdept_r = 0) AS k_factor
FROM mapunit mu
INNER JOIN component c ON mu.mukey = c.mukey
WHERE mu.mukey IN (
    SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84(
        '{TEST_WKT}')
)
AND c.majcompflag = 'Yes'
ORDER BY c.comppct_r DESC"""


def run(label, query):
    print(f"\n--- {label} ---")
    try:
        t0 = time.time()
        resp = requests.post(
            API_URL,
            data={"REQUEST": "query", "QUERY": query, "FORMAT": "JSON+COLUMNNAME"},
            timeout=TIMEOUT,
        )
        elapsed = time.time() - t0
        print(f"  HTTP status : {resp.status_code}")
        print(f"  Response ms : {elapsed*1000:.0f}")
        print(f"  Body length : {len(resp.text)} chars")

        if resp.status_code != 200:
            print(f"  ERROR body  : {resp.text[:300]}")
            return False

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print(f"  Raw body: {resp.text[:300]}")
            return False

        rows = data.get("Table", [])
        print(f"  Row count   : {len(rows)}")
        if rows:
            print(f"  Headers     : {rows[0]}")
        if len(rows) > 1:
            print(f"  First data  : {rows[1]}")
        return True

    except requests.exceptions.ConnectionError as e:
        print(f"  FAIL — ConnectionError: {e}")
    except requests.exceptions.Timeout:
        print(f"  FAIL — Timed out after {TIMEOUT}s")
    except requests.exceptions.RequestException as e:
        print(f"  FAIL — {type(e).__name__}: {e}")
    return False


def main():
    print("=" * 60)
    print("SSURGO Soil Data Access API — Diagnostic")
    print(f"Endpoint: {API_URL}")
    print("=" * 60)

    ping_ok    = run("Test 1: Endpoint ping (minimal query)", PING_QUERY)
    kfactor_ok = run("Test 2: K-factor query (Shelby County Iowa polygon)", KFACTOR_QUERY)

    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print(f"  Endpoint reachable : {'PASS' if ping_ok    else 'FAIL'}")
    print(f"  K-factor query     : {'PASS' if kfactor_ok else 'FAIL'}")

    if not ping_ok:
        print("\nDIAGNOSIS: External — API endpoint unreachable or returning errors.")
        print("  Check: https://www.nrcs.usda.gov/resources/data-and-reports/web-soil-survey")
        print("  Check: https://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx")
    elif ping_ok and not kfactor_ok:
        print("\nDIAGNOSIS: API reachable but spatial query failing.")
        print("  Possible causes:")
        print("  - SDA_Get_Mukey_from_intersection_with_WktWgs84 function deprecated/renamed")
        print("  - Spatial query timeout (complex polygon — try simplifying)")
        print("  - SSURGO spatial index temporarily offline")
    else:
        print("\nDIAGNOSIS: API appears healthy — issue may be field-specific")
        print("  (complex polygon, coordinate ordering, or geometry validity).")

    print("=" * 60)
    sys.exit(0 if (ping_ok and kfactor_ok) else 1)


if __name__ == "__main__":
    main()
