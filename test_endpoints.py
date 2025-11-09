"""
Test script for the smartsearch-retrieval API endpoints.
Make sure the Flask server is running before executing this script.
"""

import requests
import json


BASE_URL = "http://localhost:5000"


def test_send_text():
    """Test the /retrieval/send-text endpoint"""
    print("\n" + "=" * 60)
    print("Testing /retrieval/send-text endpoint")
    print("=" * 60)

    url = f"{BASE_URL}/retrieval/send-text"

    # Test case 1: Valid text
    print("\n[Test 1] Valid text input:")
    payload = {"text": "A cat sitting on a couch"}

    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Embedding dimension: {len(data['embedding'])}")
        print(f"First 5 values: {data['embedding'][:5]}")
    else:
        print(f"Error: {response.json()}")

    # Test case 2: Missing text
    print("\n[Test 2] Missing text (error expected):")
    payload = {}

    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


def test_two_text_test():
    """Test the /retrieval/two-text-test endpoint"""
    print("\n" + "=" * 60)
    print("Testing /retrieval/two-text-test endpoint")
    print("=" * 60)

    url = f"{BASE_URL}/retrieval/two-text-test"

    # Test case 1: Similar texts
    print("\n[Test 1] Similar texts:")
    payload = {
        "text1": "A dog playing in the park",
        "text2": "A puppy running in a garden",
    }

    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Text 1: {data['text1']}")
        print(f"Text 2: {data['text2']}")
        print(f"Embedding 1 dimension: {len(data['embedding1'])}")
        print(f"Embedding 2 dimension: {len(data['embedding2'])}")
        print(f"Cosine Similarity: {data['cosine_similarity']:.4f}")
        print(f"Euclidean Distance: {data['euclidean_distance']:.4f}")
        print("\nInterpretation:")
        print(f"  - Similarity score closer to 1.0 means more similar")
        print(f"  - Distance score closer to 0.0 means more similar")
    else:
        print(f"Error: {response.json()}")

    # Test case 2: Dissimilar texts
    print("\n[Test 2] Dissimilar texts:")
    payload = {
        "text1": "A beautiful sunset over the ocean",
        "text2": "Mathematical equations on a blackboard",
    }

    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Text 1: {data['text1']}")
        print(f"Text 2: {data['text2']}")
        print(f"Cosine Similarity: {data['cosine_similarity']:.4f}")
        print(f"Euclidean Distance: {data['euclidean_distance']:.4f}")
    else:
        print(f"Error: {response.json()}")

    # Test case 3: Identical texts
    print("\n[Test 3] Identical texts:")
    payload = {
        "text1": "The quick brown fox jumps over the lazy dog",
        "text2": "The quick brown fox jumps over the lazy dog",
    }

    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Text 1: {data['text1']}")
        print(f"Text 2: {data['text2']}")
        print(f"Cosine Similarity: {data['cosine_similarity']:.4f}")
        print(f"Euclidean Distance: {data['euclidean_distance']:.4f}")
        print("\nNote: Identical texts should have similarity ~1.0 and distance ~0.0")
    else:
        print(f"Error: {response.json()}")

    # Test case 4: Missing parameters
    print("\n[Test 4] Missing text2 parameter (error expected):")
    payload = {"text1": "Only one text provided"}

    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


def main():
    print("=" * 60)
    print("SmartSearch Retrieval API - Endpoint Testing")
    print("=" * 60)
    print("Make sure the Flask server is running on http://localhost:5000")

    try:
        # Test if server is running
        response = requests.get(f"{BASE_URL}/")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to the server!")
        print("Please start the Flask server first using:")
        print("  python app.py")
        return

    # Run tests
    test_send_text()
    test_two_text_test()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
