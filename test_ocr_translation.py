import subprocess
import sys

# Run the main script with a video file
subprocess.run([sys.executable, "main.py", "test_video.mp4"])

# check the output
expected_results = ["Profitez", "-", "frangais", "-",
                    "faites", "Participez", "possible", "camarades"]
results = []
matches = 0
with open("output.txt", encoding="utf-8") as f:
    results = f.read().splitlines()
for word in expected_results:
    if word in results:
        matches += 1
print(f"{matches}/{len(expected_results)} succeeded!")
