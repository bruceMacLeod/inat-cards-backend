import requests
import json


def verify_deployment(backend_url, frontend_url):
    checks = {
        "Backend Health": {
            "url": f"{backend_url}/api/data",
            "expected": {"message": "Hello from Flask!"}
        },
        "CSV Files List": {
            "url": f"{backend_url}/list_csv_files",
            "expected_type": "array"
        },
        "CORS Headers": {
            "url": f"{backend_url}/api/data",
            "headers_to_check": ["Access-Control-Allow-Origin"]
        }
    }

    results = []
    for check_name, check_info in checks.items():
        try:
            response = requests.get(check_info["url"])
            if "expected" in check_info:
                assert response.json() == check_info["expected"]
            if "expected_type" in check_info:
                if check_info["expected_type"] == "array":
                    assert isinstance(response.json()["files"], list)
            if "headers_to_check" in check_info:
                for header in check_info["headers_to_check"]:
                    assert header in response.headers
            results.append(f"✅ {check_name}: Pass")
        except Exception as e:
            results.append(f"❌ {check_name}: Fail - {str(e)}")

    return results


if __name__ == "__main__":
#    backend_url = "https://your-eb-domain.com"
#    frontend_url = "https://your-amplify-domain.com"
    backend_url = "http://localhost:8000"
    frontend_url = "http://localhost:3000"
    results = verify_deployment(backend_url, frontend_url)
    for result in results:
        print(result)