import random
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style, init

init(autoreset=True)

def load_data():
    try:
        df = pd.read_csv("imdb_top_1000.csv")
        df = df.rename(columns={
            'Series_Title': 'Title', 'title': 'Title',
            'Genre': 'Genre', 'genre': 'Genre',
            'Overview': 'Overview', 'overview': 'Overview',
            'IMDB_Rating': 'IMDB_Rating', 'imdb_rating': 'IMDB_Rating'
        })
        df = df.dropna(subset=['Title', 'Genre', 'Overview', 'IMDB_Rating'])
        df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
        df = df.dropna(subset=['IMDB_Rating'])
        df['Features'] = df['Genre'].astype(str) + " " + df['Overview'].astype(str)
        return df
    except FileNotFoundError:
        print(f"{Fore.RED}Error: 'imdb_top_1000.csv' not found.")
        return None

def extract_unique_genres(df):
    all_genres = set()
    for genre_string in df['Genre'].dropna():
        genres = [g.strip() for g in genre_string.split(',')]
        all_genres.update(genres)
    return sorted(list(all_genres))

def show_movie(row):
    polarity = TextBlob(row['Overview']).sentiment.polarity
    if polarity > 0.1:
        sent, color = "Positive", Fore.GREEN
    elif polarity < -0.1:
        sent, color = "Negative", Fore.RED
    else:
        sent, color = "Neutral", Fore.YELLOW
    
    print(f"{Fore.CYAN}MOVIE DETAILS")
    print(f"{Fore.MAGENTA}Title:       {Fore.WHITE}{row['Title']}")
    print(f"{Fore.MAGENTA}Genre(s):     {Fore.WHITE}{row['Genre']}")
    print(f"{Fore.MAGENTA}IMDb Rating:  {Fore.YELLOW}{row['IMDB_Rating']}/10")
    print(f"{Fore.MAGENTA}Sentiment:    {color}{sent}")
    print(f"{Fore.MAGENTA}Overview:")
    print(f"{Fore.LIGHTBLACK_EX}{row['Overview']}")

def get_ai_rec(df, genre, mood):
    filtered = df.copy()
    if genre:
        filtered = filtered[filtered['Genre'].str.contains(genre, case=False, na=False)]
    if filtered.empty:
        filtered = df.copy()
        
    filtered = filtered.reset_index(drop=True)
    vec = TfidfVectorizer(stop_words='english')
    matrix = vec.fit_transform(filtered['Features'])
    user_vec = vec.transform([mood])
    scores = cosine_similarity(user_vec, matrix).flatten()
    return filtered.iloc[scores.argmax()]

def main():
    df = load_data()
    if df is None: 
        return
    
    unique_genres = extract_unique_genres(df)
    
    name = input(f"{Fore.CYAN}Enter your name: ").strip()
    print(f"{Fore.GREEN}Hello {name}!")
    
    while True:
        print(f"{Fore.YELLOW}\nChoose (1) AI Recommendation (2) Random Recommendation:")
        choice = input(f"{Fore.CYAN}Enter selection: ").strip()
        
        if choice == '1':
            print(f"{Fore.MAGENTA}\nAvailable Genres:")
            print(f"{Fore.WHITE}, ".join(unique_genres))
            
            while True:
                genre = input(f"{Fore.CYAN}\nChoose a genre from the list above (or press Enter to skip): ").strip()
                if not genre or any(g.lower() == genre.lower() for g in unique_genres):
                    break
                print(f"{Fore.RED}Invalid genre. Please select an option from the printed list.")
                
            mood = input(f"{Fore.CYAN}Describe your mood or what you want to watch: ").strip()
            rec = get_ai_rec(df, genre, mood)
        elif choice == '2':
            rec = df.sample(n=1).iloc[0]
        else:
            print(f"{Fore.RED}Invalid choice.")
            continue
            
        if rec is not None: 
            show_movie(rec)
        else: 
            print(f"{Fore.RED}No matching movies found.")
        
        repeat = input(f"{Fore.YELLOW}Want another recommendation? (yes/no): ").strip().lower()
        if repeat not in ['yes', 'y']:
            print(f"{Fore.GREEN}Goodbye, {name}!")
            break

if __name__ == "__main__":
    main()