# Steam Game Recommendation System Using Neural Collaborative Filtering

This project aims to build a personalized game recommendation system for Steam users, employing Neural Collaborative Filtering (NCF) as the underlying algorithm. The model is trained on user-item interactions and leverages playtime as an implicit proxy for user preferences.

## Table of Contents

- [Background](#background)
- [Project Description](#project-description)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Work](#future-work)
- [Sources](#sources)

## Background

Recommender systems have become indispensable in a variety of domains, including e-commerce, entertainment, and game platforms. While traditional methods like collaborative filtering have shown effectiveness, deep learning models like Neural Collaborative Filtering have started to gain attention for their capability to offer more personalized and accurate recommendations. This project aims to bring these advancements to Steam, a popular gaming platform.

## Project Description

The project focuses on:

- **Custom Data Preparation**: User-item interactions are meticulously processed to create mappings and datasets that are optimal for training deep learning models.
- **Use of Playtime as Implicit Ratings**: Instead of explicit ratings, user playtime is used as an implicit signal to train the model, allowing for nuanced recommendations.
- **Model Evaluation**: Several metrics, including Precision@K and Recall@K, are used to evaluate the model's performance.
- **Scalability**: The model is designed to handle large datasets efficiently, making it suitable for real-world applications.


## Dataset Overview

The dataset used in this project is obtained from the UCSD Steam Video Game and Bundle Data collection. This dataset is expansive and includes various metrics that are relevant for building a personalized game recommendation system.

### Basic Statistics

- **Reviews**: 7,793,069
- **Users**: 2,567,538
- **Items**: 15,474
- **Bundles**: 615

### Metadata

The dataset includes metadata such as:

- **Reviews**: Includes purchase history, playtime, and recommendations ("likes").
- **Product Bundles**: Information about game bundles including pricing and bundled items.
- **Pricing Information**: Detailed pricing information for individual games and bundles.

### Example Data Points

Here's an example of a bundle data point:

```json
{
  "bundle_final_price": "$29.66",
  "bundle_url": "http://store.steampowered.com/bundle/1482/?utm_source=SteamDB...",
  "bundle_price": "$32.96",
  "bundle_name": "Two Tribes Complete Pack!",
  "bundle_id": "1482",
  "items": [
    {
      "genre": "Casual, Indie",
      "item_id": "38700",
      "discounted_price": "$4.99",
      "item_url": "http://store.steampowered.com/app/38700",
      "item_name": "Toki Tori"
    },
  ],
  "bundle_discount": "10%"
}
```

#### Data Source and Citations


- Self-attentive sequential recommendation, ICDM, 2018 [PDF](#)
- Item recommendation on monotonic behavior chains, RecSys, 2018 [PDF](#)
- Generating and personalizing bundle recommendations on Steam, SIGIR, 2017 [PDF](#)



### Data Preparation

The model is trained on a dataset of user-item interactions from Steam. Key features of the dataset include:

- **Content**: Data includes User IDs, Item IDs, and playtime.
- **Data Integrity**: Duplicate entries based on 'user_id' and 'item_id' are removed, and any missing values are handled appropriately.
- **Implicit Ratings through Playtime**: Playtime is used as an implicit form of rating, capturing the user's level of interest or engagement with a game. The playtime is scaled between 0 and 1 to facilitate the learning process.

## Model Architecture

### Neural Collaborative Filtering (NCF)

The NCF model is designed to learn from implicit feedback and consists of two main components:

- **Generalized Matrix Factorization (GMF)**: Efficiently captures the latent factors associated with each user and item.
- **Multi-Layer Perceptron (MLP)**: Processes the feature vectors to capture more complex user-item interactions.


Absolutely, let's make those updates to the Results section:

---

## Results

The model demonstrates promising results in recommending personalized games for Steam users. The performance evaluation metrics and the learning curves substantiate the model's effectiveness.

### Evaluation Metrics

- **RMSE**: The model reached an RMSE score of 0.013998393771264486 after 20 epochs, indicating a high degree of accuracy in predicting user-item interactions based on playtime.
  
- **Precision@K and Recall@K**: These metrics further validate the model's capacity to offer accurate and comprehensive game recommendations to the users.

### Learning Curves

Below are the learning curves that depict the model's performance over the training epochs:

- **Validation Curve**

  ![Validation Curve]((https://github.com/parkermoe/steam_recommendation_system/blob/main/W%26B%20Chart%2010_5_2023%2C%203_48_52%20PM_loss.png))

- **RMSE Curve**

  ![RMSE Curve]((https://github.com/parkermoe/steam_recommendation_system/blob/main/W%26B%20Chart%20RMSE%20Loss.png))

### Live Demo

You can also experience the model's recommendation capabilities firsthand by trying it out on [HuggingFace Spaces](https://huggingface.co/spaces/DimensionDweller/Steam_recsys).



## Future Work and Conclusion

This project presents a robust solution for personalized game recommendations on Steam using Neural Collaborative Filtering. Future avenues for research may include:

- Experimenting with different neural architectures and loss functions.
- Extending the model to include additional contextual features like game genres, user profiles, and seasonal trends.

## Sources

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). **Neural Collaborative Filtering**. In Proceedings of the 26th International Conference on World Wide Web (WWW).

