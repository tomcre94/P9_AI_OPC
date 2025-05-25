# utils/visualization.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def create_text_length_histogram(df, text_column='text'):
    """Crée un histogramme interactif de la longueur des textes"""
    df['text_length'] = df[text_column].apply(len)
    
    fig = px.histogram(
        df, 
        x='text_length',
        nbins=50,
        title='Distribution de la longueur des textes',
        labels={'text_length': 'Nombre de caractères', 'count': 'Nombre de textes'},
        color_discrete_sequence=['#3366CC']  # Couleur accessible
    )
    
    # Ajouter des annotations pour l'accessibilité
    fig.update_layout(
        title={
            'text': 'Distribution de la longueur des textes',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title={'text': 'Nombre de caractères', 'font': {'size': 16}},
        yaxis_title={'text': 'Nombre de textes', 'font': {'size': 16}},
        font=dict(size=14),
        template="plotly_white"
    )
    
    # Ajouter des statistiques comme annotations
    mean_length = df['text_length'].mean()
    median_length = df['text_length'].median()
    
    fig.add_annotation(
        x=mean_length,
        y=0,
        text=f"Moyenne: {mean_length:.1f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    fig.add_annotation(
        x=median_length,
        y=0,
        text=f"Médiane: {median_length:.1f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-80
    )
    
    return fig

def create_word_count_histogram(df, text_column='text'):
    """Crée un histogramme interactif du nombre de mots par texte"""
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    fig = px.histogram(
        df, 
        x='word_count',
        nbins=50,
        title='Distribution du nombre de mots par texte',
        labels={'word_count': 'Nombre de mots', 'count': 'Nombre de textes'},
        color_discrete_sequence=['#6600CC']  # Couleur accessible différente
    )
    
    # Ajouter des annotations pour l'accessibilité
    fig.update_layout(
        title={
            'text': 'Distribution du nombre de mots par texte',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title={'text': 'Nombre de mots', 'font': {'size': 16}},
        yaxis_title={'text': 'Nombre de textes', 'font': {'size': 16}},
        font=dict(size=14),
        template="plotly_white"
    )
    
    # Ajouter des statistiques comme annotations
    mean_words = df['word_count'].mean()
    median_words = df['word_count'].median()
    
    fig.add_annotation(
        x=mean_words,
        y=0,
        text=f"Moyenne: {mean_words:.1f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    fig.add_annotation(
        x=median_words,
        y=0,
        text=f"Médiane: {median_words:.1f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-80
    )
    
    return fig

def create_sentiment_distribution(df, target_column='target'):
    """Crée un graphique à barres de la distribution des sentiments"""
    sentiment_counts = df[target_column].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Définir des couleurs accessibles avec un bon contraste
    colors = ['#009E73', '#D55E00']  # Vert et orange accessibles
    
    fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        title='Distribution des sentiments dans le jeu de données',
        labels={'Count': 'Nombre d\'échantillons', 'Sentiment': 'Sentiment'},
        color_discrete_sequence=colors
    )
    
    # Ajouter des annotations pour l'accessibilité
    fig.update_layout(
        title={
            'text': 'Distribution des sentiments dans le jeu de données',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title={'text': 'Sentiment', 'font': {'size': 16}},
        yaxis_title={'text': 'Nombre d\'échantillons', 'font': {'size': 16}},
        font=dict(size=14),
        template="plotly_white"
    )
    
    # Ajouter le nombre exact sur chaque barre
    for i, row in sentiment_counts.iterrows():
        fig.add_annotation(
            x=row['Sentiment'],
            y=row['Count'],
            text=str(row['Count']),
            showarrow=False,
            yshift=10,
            font=dict(size=14)
        )
    
    return fig

def create_word_frequency_chart(df, text_column='text', top_n=20):
    """Crée un graphique à barres des mots les plus fréquents"""
    stop_words = set(stopwords.words('english'))
    
    # Concaténer tous les textes
    all_text = ' '.join(df[text_column].astype(str).tolist()).lower()
    
    # Tokeniser le texte
    words = word_tokenize(all_text)
    
    # Filtrer les mots (enlever stopwords, garder seulement les mots alphabétiques)
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Compter la fréquence des mots
    word_freq = Counter(filtered_words)
    
    # Obtenir les N mots les plus fréquents
    most_common = word_freq.most_common(top_n)
    words, counts = zip(*most_common)
    
    # Créer un DataFrame pour Plotly
    word_df = pd.DataFrame({
        'Mot': words,
        'Fréquence': counts
    })
    
    # Trier par fréquence
    word_df = word_df.sort_values('Fréquence', ascending=True)
    
    # Créer le graphique à barres horizontales
    fig = px.bar(
        word_df,
        y='Mot',
        x='Fréquence',
        orientation='h',
        title=f'Top {top_n} des mots les plus fréquents',
        color='Fréquence',
        color_continuous_scale='Viridis'
    )
    
    # Améliorer la mise en page pour l'accessibilité
    fig.update_layout(
        title={
            'text': f'Top {top_n} des mots les plus fréquents',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title={'text': 'Fréquence', 'font': {'size': 16}},
        yaxis_title={'text': 'Mot', 'font': {'size': 16}},
        font=dict(size=14),
        template="plotly_white",
        coloraxis_colorbar=dict(
            title="Fréquence",
            titlefont=dict(size=14),
            tickfont=dict(size=12)
        )
    )
    
    return fig

def create_wordcloud(df, text_column='text'):
    """Crée un nuage de mots à partir des textes"""
    stop_words = set(stopwords.words('english'))
    
    # Concaténer tous les textes
    all_text = ' '.join(df[text_column].astype(str).tolist()).lower()
    
    # Tokeniser le texte
    words = word_tokenize(all_text)
    
    # Filtrer les mots (enlever stopwords, garder seulement les mots alphabétiques)
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Rejoindre les mots filtrés
    filtered_text = ' '.join(filtered_words)
    
    # Générer le nuage de mots avec une palette de couleurs accessible
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',  # Palette de couleurs accessible
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(filtered_text)
    
    # Créer la figure matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Nuage de mots du jeu de données', fontsize=20, pad=20)
    
    return fig