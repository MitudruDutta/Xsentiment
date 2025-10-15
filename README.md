# XSentiment - US Airlines Tweet Sentiment Analysis ✈️

A comprehensive Streamlit dashboard for analyzing sentiment in tweets about US Airlines. This interactive application provides deep insights into customer sentiment, geographic distribution, and temporal patterns.

## 🌟 Features

### 📊 Sentiment Analysis
- **Overall Distribution**: View sentiment breakdown across 14,640+ tweets
  - 9,178 Negative tweets (62.7%)
  - 3,099 Neutral tweets (21.2%)
  - 2,363 Positive tweets (16.1%)
- **Visual Analytics**: Switch between histogram and pie chart visualizations
- **Real-time Metrics**: Key statistics displayed in an intuitive dashboard

### 🗺️ Geographic Insights
- **Interactive Map**: Visualize tweet locations across the USA
- **Sentiment Filtering**: Filter map by positive, neutral, or negative sentiments
- **Time-based Analysis**: Optional hourly filtering (855+ geotagged tweets)
- **Smart Filtering**: Automatically excludes invalid coordinates

### ⏰ Temporal Analysis
- **Hourly Breakdown**: Track tweet activity throughout the day
- **Sentiment Trends**: See how sentiment varies by time of day
- **Interactive Charts**: Hover to see detailed statistics

### ✈️ Airline Comparison
- **Multi-select Analysis**: Compare multiple airlines simultaneously
- **Sentiment Distribution**: Grouped bar charts for easy comparison
- **Percentage Breakdown**: Detailed sentiment percentages for each airline
- Airlines included:
  - US Airways
  - United
  - American
  - Southwest
  - Delta
  - Virgin America

### ❌ Negative Feedback Analysis
- **Top Reasons**: Identify the most common reasons for negative tweets
- **Visual Ranking**: Horizontal bar chart showing top 10 issues
- **Data-driven Insights**: Help airlines improve customer service

### ☁️ Word Cloud Visualization
- **Sentiment-specific**: Generate word clouds for positive, neutral, or negative tweets
- **Smart Text Processing**: Automatically removes URLs, mentions, and retweets
- **Color-coded**: Different color schemes for each sentiment type
- **Customizable**: Up to 150 most frequent words displayed

### 🎲 Random Tweet Sample
- **Real Examples**: View random tweets from selected sentiment category
- **Quick Insights**: Get a feel for actual customer feedback

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/MitudruDutta/Xsentiment.git
cd Xsentiment
```

2. **Install dependencies**
```bash
pip install streamlit pandas numpy plotly wordcloud matplotlib
```

3. **Ensure data file exists**
- Place `Tweets.csv` in the project directory

## 💻 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
XSentiment/
│
├── app.py              # Main Streamlit application
├── Tweets.csv          # Dataset (14,640 tweets)
└── README.md           # Project documentation
```

## 📊 Dataset Information

- **Total Tweets**: 14,640
- **Time Period**: Various timestamps
- **Airlines**: 6 major US carriers
- **Geotagged**: 1,019 tweets with coordinates (855 valid)
- **Columns**:
  - `tweet_id`: Unique identifier
  - `airline_sentiment`: positive/neutral/negative
  - `airline`: Airline name
  - `text`: Tweet content
  - `tweet_created`: Timestamp
  - `tweet_coord`: Geographic coordinates
  - `negativereason`: Reason for negative sentiment
  - And more...

## 🎨 Key Improvements

1. **Enhanced UI/UX**
   - Wide layout for better visualization
   - Emoji indicators for quick recognition
   - Color-coded sentiments (Green/Blue/Red)
   - Responsive design

2. **Better Data Handling**
   - Error handling for missing data
   - Validation for coordinates
   - Efficient caching with `@st.cache_data`
   - Smart text preprocessing

3. **Advanced Visualizations**
   - Interactive Plotly charts
   - Grouped comparisons
   - Time series analysis
   - Geographic heatmap

4. **Performance Optimization**
   - Data caching for faster load times
   - Conditional rendering
   - Efficient filtering operations

5. **User Control**
   - Toggle visibility of sections
   - Multiple filter options
   - Customizable views
   - Raw data access

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly Express**: Interactive visualizations
- **WordCloud**: Text visualization
- **Matplotlib**: Additional plotting capabilities
- **NumPy**: Numerical operations

## 📈 Use Cases

- **Customer Service**: Identify common complaints and areas for improvement
- **Market Research**: Compare airline performance and customer satisfaction
- **Trend Analysis**: Track sentiment changes over time
- **Geographic Insights**: Understand regional sentiment patterns
- **Communication Strategy**: Tailor messaging based on customer feedback

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Mitudru Dutta**
- GitHub: [@MitudruDutta](https://github.com/MitudruDutta)
- Repository: [Xsentiment](https://github.com/MitudruDutta/Xsentiment)

## 🙏 Acknowledgments

- Dataset: US Airlines Twitter Sentiment Dataset
- Built with Streamlit for rapid prototyping
- Inspired by data-driven customer service improvement

---

**Note**: This dashboard is designed for educational and analytical purposes. Insights should be combined with other data sources for business decisions.

