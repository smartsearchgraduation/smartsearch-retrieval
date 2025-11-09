from flask import Flask, request, jsonify
from services.retrieval_service import generate_embeddings, compare_text_embeddings
from utils.logger import logger

app = Flask(__name__)


@app.route("/retrieval/send-text", methods=["POST"])
def send_text():
    try:
        data = request.json
        text = data.get("text")
        if not text:
            return jsonify({"error": "Missing 'text' in request"}), 400

        embeddings = generate_embeddings([text])
        # convert to list for JSON serialization
        embeddings_list = embeddings[0].tolist()
        return jsonify({"embedding": embeddings_list})

    except Exception as e:
        logger.exception("Error in /retrieval/send-text")
        return jsonify({"error": str(e)}), 500


@app.route("/retrieval/two-text-test", methods=["POST"])
def two_text_test():
    try:
        data = request.json
        text1 = data.get("text1")
        text2 = data.get("text2")

        # Validate inputs
        if not text1 or not text2:
            return jsonify({"error": "Missing 'text1' or 'text2' in request"}), 400

        # Use the service layer to compare texts
        result = compare_text_embeddings(text1, text2)

        # Convert embeddings to lists for JSON serialization
        response = {
            "text1": text1,
            "text2": text2,
            "embedding1": result["embedding1"].tolist(),
            "embedding2": result["embedding2"].tolist(),
            "cosine_similarity": result["cosine_similarity"],
            "euclidean_distance": result["euclidean_distance"],
        }

        logger.info(
            f"Compared two texts with similarity: {result['cosine_similarity']:.4f}"
        )
        return jsonify(response)

    except Exception as e:
        logger.exception("Error in /retrieval/two-text-test")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
