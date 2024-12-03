from helper_functions import process_text  # Or wherever this function resides
# Prepare features and labels
import joblib
import numpy as np

# Load the model
model = joblib.load("model/saved_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
print("Model and feature columns loaded successfully.")

def predict_new_text(text, model, feature_columns):
    # Calculate metrics for the new text
    metrics = process_text(text)  # External function for text processing
    metrics_values = [metrics[col] for col in feature_columns]
    prediction_probabilities = model.predict_proba([metrics_values])[0]
    prediction = model.predict([metrics_values])[0]
    
    return {
        "prediction": "AI" if prediction == 0 else "Human",  # 0 or 1
        "probability_AI_generated": prediction_probabilities[1],
        "probability_human_written": prediction_probabilities[0],
    }

text="""
Globalisation: Bridging the World

Globalisation is the process by which the world becomes increasingly interconnected through trade, technology, culture, and communication. Over the past few decades, this phenomenon has accelerated, shaping societies, economies, and politics. While globalisation brings opportunities for growth and collaboration, it also poses challenges that require careful consideration.

The Driving Forces of Globalisation
Globalisation is driven by advancements in technology, transportation, and communication. The internet and digital networks have revolutionized the way information is shared, enabling instant communication across borders. Similarly, improvements in transportation, such as faster shipping and affordable air travel, have facilitated the movement of goods and people globally. Trade agreements and economic policies aimed at reducing barriers have further encouraged global interconnectedness.

Benefits of Globalisation
One of the most significant advantages of globalisation is economic growth. It has allowed countries to specialize in their strengths, leading to increased efficiency and productivity. Businesses can access global markets, and consumers benefit from a wider range of products at lower prices. For developing nations, globalisation has brought foreign investments, technology, and job opportunities, helping to lift millions out of poverty.

Cultural exchange is another benefit. Through globalisation, people are exposed to different ideas, traditions, and lifestyles, fostering greater understanding and tolerance. Popular culture, cuisine, and fashion from various parts of the world have become shared experiences, enriching societies.

Challenges of Globalisation
Despite its benefits, globalisation has drawbacks. Economic inequality has widened as wealth becomes concentrated in certain regions and among a select few. Developing countries often face exploitation of their resources and labor, with profits disproportionately benefiting multinational corporations.

Cultural homogenization is another concern. As global brands and media dominate, local traditions and languages risk being overshadowed, leading to a loss of cultural diversity. Furthermore, globalisation has contributed to environmental degradation through increased industrial activity, deforestation, and pollution.

Geopolitical tensions have also intensified. As nations become interdependent, disputes over trade, resources, and intellectual property can escalate, impacting global stability.

Balancing Globalisation
To harness the potential of globalisation while mitigating its downsides, a balanced approach is necessary. Policies should promote fair trade, protect labor rights, and ensure environmental sustainability. Encouraging cultural preservation alongside global integration can help maintain diversity.

Conclusion
Globalisation is a transformative force that has reshaped the modern world. While it offers unparalleled opportunities for connection and growth, it also requires collective responsibility to address its challenges. By fostering equitable and sustainable practices, humanity can ensure that globalisation benefits all, creating a more interconnected yet inclusive future."""
result = predict_new_text(text,model,feature_columns)

# Print the results with two decimal places for float values
for key, value in result.items():
    if isinstance(value, (np.float64, float)):
        print(f"{key}: {value:.2f}")
    elif isinstance(value, (np.int64, int)):
        print(f"{key}: {value}")