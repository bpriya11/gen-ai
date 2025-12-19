import requests
import time

CLIENT_ID = os.environ.get("CLIENT_ID")  # from your OAuth app

# Step 1: Get device/user codes
resp = requests.post(
    "https://github.com/login/device/code",
    data={"client_id": CLIENT_ID, "scope": "read:user"},
    headers={"Accept": "application/json"}
)
resp.raise_for_status()
data = resp.json()
INTERVAL = data.get("interval", 5)

print(f"👉 Visit: {data['verification_uri']} and enter the code: {data['user_code']}")

# Step 2: Poll for the access token
while True:
    token_resp = requests.post(
        "https://github.com/login/oauth/access_token",
        data={
            "client_id": CLIENT_ID,
            "device_code": data["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
        },
        headers={"Accept": "application/json"}
    )
    token_data = token_resp.json()

    if "access_token" in token_data:
        print(f"\n✅ Your GitHub OAuth Token:\n{token_data['access_token']}")
        break
    elif token_data.get("error") == "authorization_pending":
        print("⏳ Waiting for user to authorize...",token_data)
    elif token_data.get("error") == "slow_down":
        print("⚠️ Slow down! Increasing interval.")
        INTERVAL += 5
    elif token_data.get("error") == "expired_token":
        print("❌ Device code expired. Start again.")
        break
    else:
        print("❌ Unknown error:", token_data)
        break

    time.sleep(INTERVAL)
