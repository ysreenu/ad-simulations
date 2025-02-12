from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import uuid
import csv

app = Flask(__name__)
CORS(app)

class AdSimulation:
    def __init__(self):
        # Parameters
        self.K = 3  # Number of ads
        self.true_ctr = np.random.rand(self.K)  # True CTRs (unknown in practice)
        self.true_reward_mean = np.random.randint(1, 10, size=self.K)  # True reward means
        self.true_reward_std = np.random.rand(self.K)  # True reward standard deviations

        # Initialize Beta and Gaussian parameters
        self.alpha = np.ones(self.K)  # Beta distribution parameters for CTR
        self.beta = np.ones(self.K)
        self.reward_mean = np.zeros(self.K)  # Gaussian distribution parameters for reward
        self.reward_std = np.ones(self.K)

        # Track clicks, impressions, and rewards
        self.clicks = np.zeros(self.K)
        self.impressions = np.zeros(self.K)
        self.total_rewards = np.zeros(self.K)

        # CSV file to store interactions
        self.CSV_FILE = "ad_interactions.csv"

        # Initialize CSV file with headers
        with open(self.CSV_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["User ID", "Ad ID", "Click", "Reward", "Alpha", "Beta", "Reward Mean", "Reward Std"])

    def update_distributions(self, ad_index, click, reward, user_id):
        # Update Beta distribution (CTR)
        self.alpha[ad_index] += click
        self.beta[ad_index] += (1 - click)

        # Update Gaussian distribution (reward)
        n = self.clicks[ad_index]
        old_mean = self.reward_mean[ad_index]
        self.reward_mean[ad_index] = (old_mean * n + reward) / (n + 1)
        self.reward_std[ad_index] = np.sqrt(
            (self.reward_std[ad_index] ** 2 * n + (reward - old_mean) * (reward - self.reward_mean[ad_index])) / (n + 1)
        )

        # Update metrics
        self.clicks[ad_index] += click
        self.impressions[ad_index] += 1
        self.total_rewards[ad_index] += reward

        # Log interaction in CSV
        with open(self.CSV_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [user_id, ad_index + 1, click, reward, self.alpha[ad_index], self.beta[ad_index], self.reward_mean[ad_index], self.reward_std[ad_index]]
            )

    def thompson_sampling(self):
        # Sample CTRs from the Beta distribution
        sampled_ctr = np.random.beta(self.alpha, self.beta)
        # Select the ad with the highest sampled CTR
        return np.argmax(sampled_ctr)

# Create an instance of AdSimulation
ad_simulation = AdSimulation()

@app.route("/api/ads", methods=["GET"])
def get_ads():
    # Return the number of ads
    return jsonify({"ads": list(range(ad_simulation.K))})

@app.route("/api/select_ad", methods=["GET"])
def select_ad():
    # Use Thompson Sampling to select an ad
    selected_ad = ad_simulation.thompson_sampling()
    selected_ad = int(selected_ad)
    response = jsonify({"ad_id": selected_ad + 1})
    response.headers.add("Access-Control-Allow-Origin","*");
    return response

@app.route("/api/click_ad/<int:ad_index>", methods=["POST"])
def click_ad(ad_index):
    # Simulate user click and reward
    click = np.random.binomial(1, ad_simulation.true_ctr[ad_index])
    reward = np.random.normal(ad_simulation.true_reward_mean[ad_index], ad_simulation.true_reward_std[ad_index]) if click else 0

    # Generate a unique user ID for this interaction
    user_id = str(uuid.uuid4())

    # Update distributions and log interaction
    ad_simulation.update_distributions(ad_index, click, reward, user_id)

    # Return response
    return jsonify({
        "ad_id": ad_index + 1,
        "click": click,
        "reward": reward,
        "user_id": user_id
    })

@app.route("/api/stats", methods=["GET"])
def get_stats():
    # Return statistics for A/B testing
    stats = {
        "clicks": ad_simulation.clicks.tolist(),
        "impressions": ad_simulation.impressions.tolist(),
        "total_rewards": ad_simulation.total_rewards.tolist(),
        "ctr": (ad_simulation.clicks / np.where(ad_simulation.impressions == 0, 1, ad_simulation.impressions)).tolist(),
    }
    return jsonify(stats)

if __name__ == "__main__":
    app.run(debug=True)