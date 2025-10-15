import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Sentiment Analysis of US Airlines Tweets", layout="wide")

st.title("Sentiment Analysis of Tweets about US Airlines ✈️")
st.sidebar.title("Sentiment Analysis Dashboard")

st.markdown("This application analyzes the sentiment of tweets about US Airlines 🐦")
st.sidebar.markdown("Explore tweet sentiments, locations, and trends")

DATA_URL = "Tweets.csv"


@st.cache_data(persist=True)
def load_data():
    """Load and preprocess the tweets data"""
    try:
        data = pd.read_csv(DATA_URL)
        
        # Convert tweet creation time to datetime
        data['tweet_created'] = pd.to_datetime(data['tweet_created'], errors='coerce')
        
        # Parse coordinates from tweet_coord column
        data['tweet_coord'] = data['tweet_coord'].astype(str)
        coords = data['tweet_coord'].str.strip('[]').str.split(',', expand=True)
        data['lat'] = pd.to_numeric(coords[0], errors='coerce')
        data['lon'] = pd.to_numeric(coords[1], errors='coerce')
        
        # Clean text data
        data['text'] = data['text'].fillna('')
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

# Check if data loaded successfully
if data is None or len(data) == 0:
    st.error("❌ Unable to load data. Please check if 'Tweets.csv' exists in the directory.")
    st.stop()

# Display key metrics at the top
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Tweets", f"{len(data):,}")
with col2:
    st.metric("✈️ Airlines", data['airline'].nunique())
with col3:
    negative_pct = (len(data[data['airline_sentiment'] == 'negative']) / len(data) * 100)
    st.metric("😞 Negative %", f"{negative_pct:.1f}%")
with col4:
    tweets_with_coords = ((data['lat'].notna()) & ((data['lat'] != 0.0) | (data['lon'] != 0.0))).sum()
    geo_pct = (tweets_with_coords / len(data) * 100)
    st.metric("📍 Geotagged", f"{tweets_with_coords:,}", f"{geo_pct:.1f}%")
st.markdown("---")
st.info("💡 **Dataset Info**: This dashboard analyzes 14,640 tweets about US Airlines. Only 855 tweets (5.8%) include geographic coordinates, as most users don't enable location sharing on Twitter.")

st.sidebar.subheader("🎲 Random Tweet Sample")
random_tweet = st.sidebar.radio('Select sentiment', ('positive', 'neutral', 'negative'))
filtered_data = data.query('airline_sentiment == @random_tweet')[["text"]]
if len(filtered_data) > 0:
    st.sidebar.markdown(filtered_data.sample(n=1).iat[0,0])
else:
    st.sidebar.warning(f"No {random_tweet} tweets found.")


st.sidebar.markdown("### 📊 Sentiment Distribution")
select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if not st.sidebar.checkbox("Hide", True):
    st.markdown("### 📊 Overall Sentiment Distribution")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("😊 Positive", f"{sentiment_count[sentiment_count['Sentiment']=='positive']['Tweets'].values[0]:,}")
    with col2:
        st.metric("😐 Neutral", f"{sentiment_count[sentiment_count['Sentiment']=='neutral']['Tweets'].values[0]:,}")
    with col3:
        st.metric("😞 Negative", f"{sentiment_count[sentiment_count['Sentiment']=='negative']['Tweets'].values[0]:,}")
    
    # Show visualization
    if select == 'Histogram':
        fig = px.bar(
            sentiment_count, 
            x='Sentiment', 
            y='Tweets', 
            color='Sentiment',
            color_discrete_map={'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'},
            height=500,
            title="Tweet Count by Sentiment"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.pie(
            sentiment_count, 
            names='Sentiment', 
            values='Tweets',
            color='Sentiment',
            color_discrete_map={'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'},
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


st.sidebar.subheader("� Temporal Analysis")
analysis_type = st.sidebar.radio("Select analysis type", ['Timeline View', 'Geographic Map'], key='analysis_type')

if analysis_type == 'Timeline View':
    sentiment_choice = st.sidebar.selectbox('Select sentiment', ['All', 'positive', 'neutral', 'negative'], key='timeline_sentiment')
    
    if not st.sidebar.checkbox("Close", True, key='2'):
        st.markdown("### 📅 Tweet Timeline Analysis")
        
        # Filter by sentiment
        if sentiment_choice != 'All':
            timeline_data = data[data['airline_sentiment'] == sentiment_choice].copy()
        else:
            timeline_data = data.copy()
        
        # Create date and hour columns
        timeline_data['date'] = timeline_data['tweet_created'].dt.date
        timeline_data['hour'] = timeline_data['tweet_created'].dt.hour
        
        # Show overall statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Tweets", f"{len(timeline_data):,}")
        with col2:
            date_range = f"{timeline_data['date'].min()} to {timeline_data['date'].max()}"
            st.metric("📆 Date Range", f"{(timeline_data['date'].max() - timeline_data['date'].min()).days} days")
        with col3:
            avg_per_day = len(timeline_data) / max((timeline_data['date'].max() - timeline_data['date'].min()).days, 1)
            st.metric("📈 Avg per Day", f"{avg_per_day:.0f}")
        
        # Timeline by date
        st.markdown("#### Tweets Over Time")
        daily_counts = timeline_data.groupby(['date', 'airline_sentiment']).size().reset_index(name='count')
        fig_timeline = px.line(
            daily_counts,
            x='date',
            y='count',
            color='airline_sentiment',
            color_discrete_map={'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'},
            labels={'date': 'Date', 'count': 'Number of Tweets', 'airline_sentiment': 'Sentiment'},
            title='Daily Tweet Volume by Sentiment',
            markers=True
        )
        fig_timeline.update_layout(hovermode='x unified', height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Hourly distribution
        st.markdown("#### Hourly Distribution")
        hourly_counts = timeline_data.groupby(['hour', 'airline_sentiment']).size().reset_index(name='count')
        fig_hourly = px.bar(
            hourly_counts,
            x='hour',
            y='count',
            color='airline_sentiment',
            color_discrete_map={'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'},
            labels={'hour': 'Hour of Day', 'count': 'Number of Tweets', 'airline_sentiment': 'Sentiment'},
            title='Tweet Distribution by Hour of Day',
            barmode='stack'
        )
        fig_hourly.update_layout(height=400)
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Day of week analysis
        timeline_data['day_of_week'] = pd.to_datetime(timeline_data['tweet_created']).dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        st.markdown("#### Day of Week Patterns")
        dow_counts = timeline_data.groupby(['day_of_week', 'airline_sentiment']).size().reset_index(name='count')
        dow_counts['day_of_week'] = pd.Categorical(dow_counts['day_of_week'], categories=day_order, ordered=True)
        dow_counts = dow_counts.sort_values('day_of_week')
        
        fig_dow = px.bar(
            dow_counts,
            x='day_of_week',
            y='count',
            color='airline_sentiment',
            color_discrete_map={'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'},
            labels={'day_of_week': 'Day of Week', 'count': 'Number of Tweets', 'airline_sentiment': 'Sentiment'},
            title='Tweet Distribution by Day of Week',
            barmode='group'
        )
        fig_dow.update_layout(height=400)
        st.plotly_chart(fig_dow, use_container_width=True)
        
        if st.sidebar.checkbox("Show raw data", False, key='show_raw'):
            st.subheader("Raw Timeline Data")
            st.write(timeline_data[['tweet_created', 'airline', 'airline_sentiment', 'text']].sort_values('tweet_created', ascending=False))

else:  # Geographic Map
    sentiment_choice = st.sidebar.selectbox('Select sentiment to view on map', ['All', 'positive', 'neutral', 'negative'], key='map_sentiment')
    hour_filter = st.sidebar.checkbox("Filter by hour of day", False, key='hour_filter')
    hour = 0
    if hour_filter:
        hour = st.sidebar.slider("Hour of day", 0, 23)
    
    if not st.sidebar.checkbox("Close", True, key='2'):
        st.markdown("### 🗺️ Geographic Distribution of Tweets")
        
        # Filter by sentiment
        if sentiment_choice != 'All':
            map_data = data[data['airline_sentiment'] == sentiment_choice].copy()
        else:
            map_data = data.copy()
        
        # Filter by hour if enabled
        if hour_filter:
            map_data = map_data[map_data['tweet_created'].dt.hour == hour]
        
        # Show stats before filtering coordinates
        tweets_before_filter = len(map_data)
        
        # Filter valid coordinates (excluding NaN and [0.0, 0.0])
        map_data = map_data.dropna(subset=['lat', 'lon'])
        map_data = map_data[(map_data['lat'] != 0.0) | (map_data['lon'] != 0.0)]
        
        # Display detailed statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Total Tweets Filtered", f"{tweets_before_filter:,}")
        with col2:
            st.metric("📍 With Valid Coordinates", f"{len(map_data):,}")
        with col3:
            coverage_pct = (len(map_data) / tweets_before_filter * 100) if tweets_before_filter > 0 else 0
            st.metric("📊 Coverage", f"{coverage_pct:.1f}%")
        
        st.info(f"ℹ️ **Note**: Only {855:,} out of {14640:,} total tweets ({855/14640*100:.1f}%) include geographic coordinates. Most Twitter users don't share their location.")
        
        if len(map_data) > 0:
            st.map(map_data)
        else:
            st.warning("⚠️ No location data available for the selected filters")
        
        if st.sidebar.checkbox("Show raw data", False, key='show_raw'):
            st.subheader("Raw Data")
            st.write(map_data)
        
st.sidebar.subheader("⏰ Hourly Tweet Activity")
show_hourly = st.sidebar.checkbox("Show hourly breakdown", False, key='hourly')

if show_hourly:
    st.markdown("### ⏰ Tweets by Hour of Day")
    data['hour'] = data['tweet_created'].dt.hour
    hourly_data = data.groupby(['hour', 'airline_sentiment']).size().reset_index(name='count')
    
    fig_hourly = px.line(
        hourly_data,
        x='hour',
        y='count',
        color='airline_sentiment',
        color_discrete_map={'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'},
        labels={'hour': 'Hour of Day', 'count': 'Number of Tweets', 'airline_sentiment': 'Sentiment'},
        title='Tweet Activity Throughout the Day',
        markers=True
    )
    fig_hourly.update_layout(hovermode='x unified')
    st.plotly_chart(fig_hourly, use_container_width=True)

st.sidebar.subheader("✈️ Airline Sentiment Breakdown")
choice = st.sidebar.multiselect(
    'Select airlines to compare', 
    ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), 
    key='4'
)

if len(choice) > 0:
    st.markdown(f"### ✈️ Sentiment Distribution for Selected Airlines")
    choice_data = data[data.airline.isin(choice)]
    
    # Create a grouped bar chart
    airline_sentiment = choice_data.groupby(['airline', 'airline_sentiment']).size().reset_index(name='count')
    fig_choice = px.bar(
        airline_sentiment,
        x='airline',
        y='count',
        color='airline_sentiment',
        barmode='group',
        color_discrete_map={'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'},
        labels={'airline': 'Airline', 'count': 'Number of Tweets', 'airline_sentiment': 'Sentiment'},
        title=f'Sentiment Breakdown for {", ".join(choice)}',
        height=500
    )
    st.plotly_chart(fig_choice, use_container_width=True)
    
    # Show percentage breakdown
    st.markdown("#### Sentiment Percentages by Airline")
    for airline in choice:
        airline_data = choice_data[choice_data['airline'] == airline]
        total = len(airline_data)
        sentiments = airline_data['airline_sentiment'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**{airline}**")
        with col2:
            pos_pct = (sentiments.get('positive', 0) / total * 100) if total > 0 else 0
            st.write(f"😊 {pos_pct:.1f}%")
        with col3:
            neu_pct = (sentiments.get('neutral', 0) / total * 100) if total > 0 else 0
            st.write(f"😐 {neu_pct:.1f}%")
        with col4:
            neg_pct = (sentiments.get('negative', 0) / total * 100) if total > 0 else 0
            st.write(f"😞 {neg_pct:.1f}%")


st.sidebar.subheader("❌ Negative Tweet Reasons")
show_reasons = st.sidebar.checkbox("Show negative reasons breakdown", False, key='reasons')

if show_reasons:
    st.markdown("### ❌ Top Reasons for Negative Sentiment")
    negative_reasons = data[data['airline_sentiment'] == 'negative']['negativereason'].value_counts().head(10)
    negative_reasons_df = pd.DataFrame({'Reason': negative_reasons.index, 'Count': negative_reasons.values})
    
    fig_reasons = px.bar(
        negative_reasons_df,
        x='Count',
        y='Reason',
        orientation='h',
        color='Count',
        color_continuous_scale='Reds',
        title='Top 10 Reasons for Negative Tweets',
        height=500
    )
    fig_reasons.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_reasons, use_container_width=True)

st.sidebar.subheader("☁️ Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))

if not st.sidebar.checkbox("Close", True, key='3'):
    st.header(f"☁️ Word Cloud for {word_sentiment.upper()} Sentiment")
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    
    if len(processed_words.strip()) > 0:
        # Select colormap based on sentiment
        colormap = {
            'positive': 'Greens',
            'neutral': 'Blues',
            'negative': 'Reds'
        }
        
        wordcloud = WordCloud(
            stopwords=STOPWORDS, 
            background_color='white', 
            height=640, 
            width=800,
            max_words=150,
            colormap=colormap.get(word_sentiment, 'viridis'),
            relative_scaling=0.5,
            min_font_size=10
        ).generate(processed_words)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No words available to generate word cloud.")