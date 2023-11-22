import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import datetime

# Spotify API credentials

# Custom CSS to hide header and footer
custom_css = """
<style>
    /* Hide the Streamlit header */
    header {
        display: none;
    }

    /* Hide the Streamlit footer */
    footer {
        display: none;
    }
</style>
"""

# Display custom CSS in Streamlit
st.markdown(custom_css, unsafe_allow_html=True)
st.title("Spotify Playlist Analyser")
st.caption("A fun tool to analyse your playlist")

# Input for user's playlist link
my_playlist_link = st.text_input("Enter your Spotify playlist link:")

# Input for friend's playlist link
friend_playlist_link = st.text_input("Enter your friend's Spotify playlist link:")
genplay =  st.button("Analyse Playlists")

SPOTIPY_CLIENT_ID = "0deba55448434c858c6e13250d810f77"
SPOTIPY_CLIENT_SECRET = "964718b5005a43629cdd63be7402fea0"
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                          client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def fetch_playlist_tracks(playlist_link):
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    column_list= ['uri','track_name','track_popularity', 'added_at', 'artist_name', 'artist_popularity', 'genre', 'artist_url']
    final_data = []

    for i in sp.playlist_tracks(playlist_URI)['items']:
        if i['track'] is not None:
            artist_uri = i["track"]["artists"][0]["uri"]
            artist_info = sp.artist(artist_uri)
            artist_name = i["track"]["artists"][0]["name"]
            artist_pop = artist_info["popularity"]
            artist_genres = artist_info["genres"]

            data_pos = [
            i['track']['uri'],i['track']['name'], i["track"]["popularity"], i['added_at'].split('T')[0],
            artist_name, artist_pop, ' '.join(artist_genres), i['track']['artists'][0]['external_urls']['spotify']
            ]

            final_data.append(data_pos)

    m_df = pd.DataFrame(final_data, columns = column_list)
    m_df['added_at'] = m_df['added_at'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())


    return m_df


def calculate_cosine_similarity(playlist1_df, playlist2_df):
    # Concatenate 'artist_name' and 'genre' columns, fill null values with whitespace
    playlist1_data = (playlist1_df['artist_name'].fillna('') + ' ' + playlist1_df['genre'].fillna('')).values
    playlist2_data = (playlist2_df['artist_name'].fillna('') + ' ' + playlist2_df['genre'].fillna('')).values

    m_simconcat1 = ' '.join(set(' '.join(playlist1_data).lower().split()))
    m_simconcat2 = ' '.join(set(' '.join(playlist2_data).lower().split()))

    # Create binary vectors for each playlist
    vectorizer = CountVectorizer()
    playlist1_vector = vectorizer.fit_transform([m_simconcat1]).toarray()
    playlist2_vector = vectorizer.transform([m_simconcat2]).toarray()  # Use transform instead of fit_transform for playlist2

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(playlist1_vector, playlist2_vector)
    cosine_similarity_score = similarity_matrix[0, 0]  # Get the similarity score from the matrix

    # Convert the score to percentage
    similarity_percentage = cosine_similarity_score * 100

    return round(similarity_percentage,2)

def extract_playlist_info(playlist_df, user_label):
    # Convert 'added_at' to datetime
    playlist_df['added_at'] = pd.to_datetime(playlist_df['added_at'], errors='coerce')

    # Extract oldest song
    oldest_song = playlist_df.loc[playlist_df['added_at'].idxmin()]['track_name']

    # Extract latest song
    latest_song = playlist_df.loc[playlist_df['added_at'].idxmax()]['track_name']

    # Convert 'artist_popularity' and 'track_popularity' to numeric
    playlist_df['artist_popularity'] = pd.to_numeric(playlist_df['artist_popularity'], errors='coerce')
    playlist_df['track_popularity'] = pd.to_numeric(playlist_df['track_popularity'], errors='coerce')

    # Extract most popular song
    most_popular_song = playlist_df.loc[playlist_df['track_popularity'].idxmax()]['track_name']

    # Extract most popular artist
    most_popular_artist = playlist_df['artist_name'].mode().iloc[0]

    # Extract most added artist and count
    most_added_artist_info = playlist_df['artist_name'].value_counts().idxmax()
    most_added_artist_count = playlist_df['artist_name'].value_counts().max()
    most_added_artist = f"{most_added_artist_info} ({most_added_artist_count})"

    # Create a DataFrame with the extracted information
    playlist_info_df = pd.DataFrame({
        user_label: [oldest_song, latest_song, most_popular_song, most_popular_artist, most_added_artist]
    }, index=['oldest song', 'latest song', 'most popular song', 'most popular artist', 'most added artist'])

    return playlist_info_df

def extract_date(datetime_str):
    return datetime_str.date()


def display_common_songs(df1, df2):
    # Find common songs based on 'track_name'
    
    common_songs_df = pd.merge(df1, df2, on='track_name', how='inner', suffixes=('_you', '_your_friend'))

    # Display the common songs
    if len(common_songs_df)>1:
        st.table(common_songs_df[['track_name', 'track_popularity_you', 'added_at_you', 'artist_name_you',
                              'artist_popularity_you', 'genre_you', 'artist_url_you']])
    else:
        st.warning('No common songs found!')




if genplay:
    with st.spinner('Analysing...'):
        tlink1 = fetch_playlist_tracks(my_playlist_link)
        tlink2 = fetch_playlist_tracks(friend_playlist_link)

        with st.expander('Your playlist'):
            st.dataframe(tlink1)
        
        with st.expander("Your friend's playlist"):
            st.dataframe(tlink2)

        
        similarity_score = calculate_cosine_similarity(tlink1, tlink2)


        st.header(f"Music taste similarity: {similarity_score}%")
        st.caption('It shows the music taste similarity between you and your friend!')


        st.header('Stats')
        st.caption('Some cool stats based on the playlists')

        you_playlist_info = extract_playlist_info(tlink1, 'you')
        friend_playlist_info = extract_playlist_info(tlink2, 'your friend')

        col3, col4 = st.columns(2)
        with col3:

            st.dataframe(you_playlist_info, use_container_width=True)
        with col4:

            st.dataframe(friend_playlist_info, use_container_width = True)


        # Extract dates and count for both playlists
        you_dates = tlink1['added_at'].apply(extract_date)
        friend_dates = tlink2['added_at'].apply(extract_date)

        # Count occurrences of each date
        you_dates_counts = you_dates.value_counts()
        friend_dates_counts = friend_dates.value_counts()

        # Combine counts into a DataFrame
        data = {
            'you': you_dates_counts,
            'your friend': friend_dates_counts
        }

        date_counts_df = pd.DataFrame(data)

        st.caption('Songs added at dates')
        # Plot the bar chart using Streamlit
        st.bar_chart(date_counts_df)

        st.write('Common songs')
        display_common_songs(tlink1, tlink2)

        

