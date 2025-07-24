import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Sample dataset
data = {
    'title': [
        'The Matrix', 'Inception', 'Interstellar',
        'The Dark Knight', 'Batman Begins', 'The Prestige',
        'Avengers', 'Iron Man', 'Doctor Strange'
    ],
    'description': [
        'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
        'A thief who steals corporate secrets through use of dream-sharing technology.',
        'A team travels through a wormhole in space in an attempt to ensure humanity’s survival.',
        'Batman faces a psychopathic criminal known as the Joker who wants to watch the world burn.',
        'After training with his mentor, Batman begins his fight to free Gotham from crime.',
        'Two magicians engage in a battle to create the ultimate illusion while sacrificing everything.',
        'Earth’s mightiest heroes must come together to stop a threat to global security.',
        'After being held captive, a billionaire builds a high-tech suit to escape and fight evil.',
        'A brilliant neurosurgeon learns the mystic arts to protect the world from magical threats.'
    ]
}
df = pd.DataFrame(data)

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'])

# Step 3: Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 4: Recommendation function
def recommend(movie_title, df=df, cosine_sim=cosine_sim):
    index = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Example usage: Recommend movies similar to "Inception"
recommended_movies = recommend("Iron Man")

# Print output line by line
print("Recommended movies:")
for movie in recommended_movies:
    print(movie)