import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np
from dotenv import load_dotenv
from pdf_report_generator import (
    NomadiaMonthlyReport,
    MultiMonthReportGenerator,
    generate_monthly_report_streamlit,
    get_available_months,
    validate_month
)

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Nomadia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-critical {
        background-color: #fee;
        border-left: 4px solid #e74c3c;
    }
    .alert-warning {
        background-color: #fef5e7;
        border-left: 4px solid #f39c12;
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()

# Configuration Airtable
AIRTABLE_TOKEN = os.getenv('AIRTABLE_TOKEN')
BASE_ID = 'appkOBTB6yZjdHbm5'
TABLE_NAME = 'Signalements'


# Headers pour les requêtes API
headers = {
    'Authorization': f'Bearer {AIRTABLE_TOKEN}',
    'Content-Type': 'application/json'
}

@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def fetch_airtable_data():
    """Récupère les données depuis Airtable"""
    url = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}'
    
    all_records = []
    params = {}
    
    try:
        while True:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            all_records.extend(data.get('records', []))
            
            # Pagination
            if 'offset' in data:
                params['offset'] = data['offset']
            else:
                break
        
        return all_records
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return []

def process_data(records):
    """Transforme les données Airtable en DataFrame"""
    if not records:
        return pd.DataFrame()
    
    data = []
    for record in records:
        fields = record.get('fields', {})
        
        # Extraction et nettoyage des données
        row = {
            'ID': fields.get('ID'),
            'Commune': ', '.join(fields.get('Commune recherche', [])) if isinstance(fields.get('Commune recherche'), list) else fields.get('Commune recherche', ''),
            'Intercommunalité': ', '.join(fields.get('Intercommunalité', [])) if isinstance(fields.get('Intercommunalité'), list) else fields.get('Intercommunalité', ''),
            'Arrondissement': ', '.join(fields.get('Arrondissement', [])) if isinstance(fields.get('Arrondissement'), list) else fields.get('Arrondissement', ''),
            'Adresse': fields.get('Adresse du stationnement', ''),
            'Date_Debut': fields.get('Date Début de stationnement'),
            'Date_Fin': fields.get('Date fin de stationnement'),
            'Menages': fields.get('Nombre de ménages', 0),
            'Caravanes': fields.get('Nombre de caravanes estimées', 0),
            'Terrain': fields.get('Statut du terrain', ''),
            'Statut_Stationnement': fields.get('Statut du stationnement', ''),
            'Etat_Gestion': fields.get('Etat de gestion du dossier', ''),
            'Situation': fields.get('Situation du voyageur', ''),
            'Gestionnaire': fields.get('Nom du gestionnaire du stationnement', ''),
            'Referent': fields.get('Référent du Groupe', ''),
            'Nb_Interventions': fields.get('Nombre d\'interventions', 0),
            'Delai_1ere_Intervention': fields.get('Délai en jours pour la première intervention'),
            'Duree_Stationnement': fields.get('Durée en jours du stationnement'),
            'Eau': fields.get('Eau'),
            'Electricite': fields.get('Electricité'),
            'Assainissement': fields.get('Assainissement'),
            'Telephone': fields.get('Numéro de téléphone', ''),
            'Email': fields.get('mail', ''),
            'Acteurs': fields.get('Acteurs Mobilisés sur la gestion du Dossier', ''),
            'Journal_Interventions': fields.get('Journal interventions', [])
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Conversion des dates
    if 'Date_Debut' in df.columns:
        df['Date_Debut'] = pd.to_datetime(df['Date_Debut'], errors='coerce')
    if 'Date_Fin' in df.columns:
        df['Date_Fin'] = pd.to_datetime(df['Date_Fin'], errors='coerce')
    
    return df

def calculate_priority_score(row):
    """Calcule un score de priorité pour chaque signalement"""
    if row['Nb_Interventions'] == 0:
        return 999  # Très haute priorité
    
    base_score = (row['Menages'] * row['Delai_1ere_Intervention']) / row['Nb_Interventions']
    
    # Bonus si toujours en cours
    if row['Etat_Gestion'] in ['Diagnostic en cours', 'A traiter', 'Interlocuteur consulté']:
        base_score *= 1.5
    
    return base_score

def main():
    # Header
    st.markdown('<p class="main-header">📊 Dashboard Nomadia - Gestion des Signalements</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=Nomadia", use_container_width=True)
        st.markdown("---")
        
        # Bouton de rafraîchissement
        if st.button("🔄 Actualiser les données", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 Filtres")
    
    # Chargement des données
    with st.spinner("Chargement des données..."):
        records = fetch_airtable_data()
        df = process_data(records)
    
    if df.empty:
        st.warning("⚠️ Aucune donnée disponible. Vérifiez votre token Airtable.")
        return
    
    # Calcul du score de priorité
    df['Score_Priorite'] = df.apply(calculate_priority_score, axis=1)
    
    # Filtres dans la sidebar
    with st.sidebar:
        # Filtre par intercommunalité
        intercommunalites = ['Toutes'] + sorted(df['Intercommunalité'].dropna().unique().tolist())
        selected_inter = st.selectbox("🏛️ Intercommunalité", intercommunalites)
        
        # Filtre par statut
        statuts = ['Tous'] + sorted(df['Etat_Gestion'].dropna().unique().tolist())
        selected_statut = st.selectbox("📌 État de gestion", statuts)
        
        # Filtre par arrondissement
        arrondissements = ['Tous'] + sorted(df['Arrondissement'].dropna().unique().tolist())
        selected_arron = st.selectbox("📍 Arrondissement", arrondissements)
        
        # Filtre par période
        st.markdown("### 📅 Période")
        if not df['Date_Debut'].isna().all():
            date_min = df['Date_Debut'].min()
            date_max = df['Date_Debut'].max()
            date_range = st.date_input(
                "Sélectionner une période",
                value=(date_min, date_max),
                min_value=date_min,
                max_value=date_max
            )
    
    # Application des filtres
    df_filtered = df.copy()
    
    if selected_inter != 'Toutes':
        df_filtered = df_filtered[df_filtered['Intercommunalité'] == selected_inter]
    
    if selected_statut != 'Tous':
        df_filtered = df_filtered[df_filtered['Etat_Gestion'] == selected_statut]
    
    if selected_arron != 'Tous':
        df_filtered = df_filtered[df_filtered['Arrondissement'] == selected_arron]
    
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['Date_Debut'] >= pd.Timestamp(date_range[0])) &
            (df_filtered['Date_Debut'] <= pd.Timestamp(date_range[1]))
        ]
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📄 Rapports & Exports")
        
        # Onglet rapport
        report_type = st.radio(
            "Type de rapport",
            ["Rapport Mensuel", "Comparaison Multi-mois"],
            key="report_type"
        )
        
        if report_type == "Rapport Mensuel":
            # Sélection du mois
            col1, col2 = st.columns(2)
            
            with col1:
                selected_month = st.selectbox(
                    "Mois",
                    range(1, 13),
                    format_func=lambda x: ["Janvier", "Février", "Mars", "Avril", 
                                        "Mai", "Juin", "Juillet", "Août",
                                        "Septembre", "Octobre", "Novembre", "Décembre"][x-1]
                )
            
            with col2:
                selected_year = st.selectbox(
                    "Année",
                    range(2020, datetime.now().year + 1),
                    index=datetime.now().year - 2020
                )
            
            # Vérification et génération
            if validate_month(selected_month, selected_year, df):
                if st.button("📄 Générer le rapport", use_container_width=True, key="gen_report"):
                    with st.spinner("Génération du rapport en cours..."):
                        try:
                            pdf_buffer = generate_monthly_report_streamlit(
                                df, selected_month, selected_year
                            )
                            
                            st.download_button(
                                label="📥 Télécharger le PDF",
                                data=pdf_buffer,
                                file_name=f"rapport_nomadia_{selected_month:02d}_{selected_year}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="download_report"
                            )
                            st.success("✅ Rapport généré avec succès!")
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération: {str(e)}")
            else:
                st.warning(f"⚠️ Aucune donnée pour {selected_month:02d}/{selected_year}")
        
        else:  # Comparaison multi-mois
            st.markdown("**Sélectionner les mois à comparer**")
            
            available_months = get_available_months(df)
            
            # Afficher les mois disponibles
            selected_months_list = []
            
            # Créer des cases à cocher pour chaque mois disponible
            month_names = {
                1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Jun",
                7: "Jul", 8: "Aoû", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
            }
            
            for month, year in sorted(available_months, reverse=True)[:6]:  # 6 derniers mois
                label = f"{month_names[month]} {year}"
                if st.checkbox(label, key=f"month_{month}_{year}"):
                    selected_months_list.append((month, year))
            
            if len(selected_months_list) >= 2:
                if st.button("🔄 Générer la comparaison", use_container_width=True, 
                            key="gen_comparison"):
                    with st.spinner("Génération du rapport comparatif..."):
                        try:
                            multi_report = MultiMonthReportGenerator(
                                df, selected_months_list
                            )
                            pdf_buffer = multi_report.generate_comparison_pdf()
                            
                            period_str = "_".join([f"{m[0]:02d}_{m[1]}" 
                                                for m in selected_months_list])
                            
                            st.download_button(
                                label="📥 Télécharger la comparaison",
                                data=pdf_buffer,
                                file_name=f"comparaison_nomadia_{period_str}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="download_comparison"
                            )
                            st.success("✅ Rapport comparatif généré!")
                            
                        except Exception as e:
                            st.error(f"❌ Erreur: {str(e)}")
            
            elif len(selected_months_list) > 0:
                st.info(f"ℹ️ Sélectionnez au moins 2 mois ({len(selected_months_list)}/2 actuellement)")

    # === ALERTES CRITIQUES ===
    st.markdown("### 📊 Activité Globale & Évolution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Nombre total de signalements avec style épuré
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; 
                        border-radius: 15px; 
                        text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        height: 250px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;'>
                <p style='color: rgba(255,255,255,0.9); 
                          font-size: 0.9rem; 
                          margin: 0;
                          font-weight: 500;
                          letter-spacing: 1px;'>SIGNALEMENTS TOTAUX</p>
                <h1 style='color: white; 
                           font-size: 4rem; 
                           margin: 0.5rem 0;
                           font-weight: 700;
                           line-height: 1;'>{}</h1>
                <p style='color: rgba(255,255,255,0.8); 
                          font-size: 0.85rem; 
                          margin: 0;'>sur la période sélectionnée</p>
            </div>
        """.format(len(df_filtered)), unsafe_allow_html=True)
    
    with col2:
        # Graphique d'évolution mensuelle épuré
        if not df_filtered['Date_Debut'].isna().all():
            df_evolution = df_filtered.copy()
            df_evolution['Mois'] = df_evolution['Date_Debut'].dt.to_period('M')
            evolution_counts = df_evolution.groupby('Mois').size().reset_index(name='Signalements')
            evolution_counts['Mois'] = evolution_counts['Mois'].astype(str)
            
            fig_evolution = go.Figure()
            
            # Courbe smooth
            fig_evolution.add_trace(go.Scatter(
                x=evolution_counts['Mois'],
                y=evolution_counts['Signalements'],
                mode='lines',
                line=dict(
                    color='#667eea',
                    width=3,
                    shape='spline'
                ),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)',
                hovertemplate='<b>%{x}</b><br>Signalements: %{y}<extra></extra>'
            ))
            
            fig_evolution.update_layout(
                title={
                    'text': 'Évolution Mensuelle des Signalements',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#333'}
                },
                xaxis=dict(
                    title='',
                    showgrid=False,
                    showline=True,
                    linecolor='rgba(0,0,0,0.1)'
                ),
                yaxis=dict(
                    title='Nombre de signalements',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)',
                    showline=False
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                showlegend=False,
                height=250,
                margin=dict(t=40, b=40, l=60, r=20)
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
        else:
            st.info("Aucune donnée de date disponible pour afficher l'évolution")
    
    st.markdown("---")
    
    # === KPIs PRINCIPAUX ===
    st.markdown("### 📈 Vue d'Ensemble")
    
    # Calculer les ménages et caravanes actuellement présents
    # (signalements en cours = pas encore de date de fin OU date de fin dans le futur)
    aujourd_hui = pd.Timestamp.now()
    dossiers_presents = df_filtered[
        (df_filtered['Date_Fin'].isna()) | (df_filtered['Date_Fin'] >= aujourd_hui)
    ]
    
    menages_presents = dossiers_presents['Menages'].sum()
    caravanes_presentes = dossiers_presents['Caravanes'].sum()
    
    # Ligne 1 : État du territoire
    st.markdown("#### 🏘️ État Actuel du Territoire")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        nb_signalements_actifs = len(dossiers_presents)
        st.metric(
            "Signalements actifs", 
            nb_signalements_actifs,
            delta=f"sur {len(df_filtered)} total"
        )
    
    with col2:
        st.metric(
            "Ménages présents", 
            int(menages_presents),
            help="Ménages actuellement sur le territoire (stationnement en cours)"
        )
    
    with col3:
        st.metric(
            "Caravanes présentes", 
            int(caravanes_presentes),
            help="Caravanes actuellement sur le territoire (stationnement en cours)"
        )
    
    with col4:
        ratio_actuel = caravanes_presentes / menages_presents if menages_presents > 0 else 0
        st.metric(
            "Caravanes/Ménage", 
            f"{ratio_actuel:.1f}",
            help="Nombre moyen de caravanes par ménage actuellement présent"
        )
    
    # Ligne 2 : Indicateurs de performance
    st.markdown("#### ⚡ Performance de Gestion")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delai_moyen = df_filtered[df_filtered['Delai_1ere_Intervention'].notna()]['Delai_1ere_Intervention'].mean()
        st.metric(
            "Délai moyen 1ère intervention", 
            f"{delai_moyen:.1f}j" if not pd.isna(delai_moyen) else "N/A",
            delta="Objectif: <7j" if delai_moyen > 7 else "✓ Objectif atteint",
            delta_color="inverse" if delai_moyen > 7 else "normal",
            help="Temps moyen entre le début du stationnement et la première intervention"
        )
    
    with col2:
        duree_moyenne = df_filtered[df_filtered['Duree_Stationnement'].notna()]['Duree_Stationnement'].mean()
        st.metric(
            "Durée moyenne de stationnement", 
            f"{duree_moyenne:.0f}j" if not pd.isna(duree_moyenne) else "N/A",
            help="Durée moyenne entre l'arrivée et le départ des groupes"
        )
    
    with col3:
        dossiers_urgents = len(df_filtered[
            (df_filtered['Delai_1ere_Intervention'] > 30) | 
            ((df_filtered['Nb_Interventions'] == 0) & (df_filtered['Etat_Gestion'] != 'Fin du stationnement'))
        ])
        pourcentage_urgents = (dossiers_urgents / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.metric(
            "Dossiers urgents", 
            dossiers_urgents,
            delta=f"{pourcentage_urgents:.0f}% du total",
            delta_color="inverse" if dossiers_urgents > 0 else "normal",
            help="Dossiers avec >30j sans intervention ou sans aucune intervention"
        )
    
    # === PERFORMANCE OPÉRATIONNELLE ===
    st.markdown("---")
    st.markdown("### ⚡ Performance Opérationnelle")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rapides = len(df_filtered[df_filtered['Delai_1ere_Intervention'] <= 7])
        taux_reactivite = (rapides / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.metric("⚡ Réactivité (<7j)", f"{taux_reactivite:.0f}%",
                 delta=f"{rapides}/{len(df_filtered)} dossiers")
    
    with col2:
        delai_moyen = df_filtered['Delai_1ere_Intervention'].mean()
        st.metric("⏱️ Délai moyen 1ère intervention", 
                 f"{delai_moyen:.1f}j" if not pd.isna(delai_moyen) else "N/A")
    
    with col3:
        nb_interv_moyen = df_filtered['Nb_Interventions'].mean()
        st.metric("🔄 Interventions moyennes/dossier", f"{nb_interv_moyen:.1f}")
    
    with col4:
        duree_moyenne = df_filtered[df_filtered['Duree_Stationnement'].notna()]['Duree_Stationnement'].mean()
        st.metric("📅 Durée moyenne de présence", 
                 f"{duree_moyenne:.0f}j" if not pd.isna(duree_moyenne) else "N/A")
    
    # Graphiques performance
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des délais
        if not df_filtered['Delai_1ere_Intervention'].isna().all():
            fig_delai = px.histogram(
                df_filtered,
                x='Delai_1ere_Intervention',
                nbins=20,
                title="Distribution des délais d'intervention (jours)",
                color_discrete_sequence=['#3498db']
            )
            fig_delai.add_vline(x=7, line_dash="dash", line_color="green", 
                               annotation_text="Objectif 7j")
            fig_delai.add_vline(x=20, line_dash="dash", line_color="orange",
                               annotation_text="Seuil 20j")
            fig_delai.update_layout(showlegend=False)
            st.plotly_chart(fig_delai, use_container_width=True)
    
    with col2:
        # Corrélation interventions / durée
        if not df_filtered['Duree_Stationnement'].isna().all():
            fig_corr = px.scatter(
                df_filtered[df_filtered['Duree_Stationnement'].notna()],
                x='Nb_Interventions',
                y='Duree_Stationnement',
                size='Menages',
                color='Etat_Gestion',
                hover_data=['Commune', 'ID'],
                title="Corrélation : Interventions vs Durée de présence",
                labels={'Nb_Interventions': 'Nombre d\'interventions',
                       'Duree_Stationnement': 'Durée (jours)'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # === ANALYSE TERRITORIALE ===
    st.markdown("---")
    st.markdown("### 🗺️ Analyse Territoriale")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top communes
        top_communes = df_filtered['Commune'].value_counts().head(5).reset_index()
        top_communes.columns = ['Commune', 'Nombre']
        
        fig_communes = px.bar(
            top_communes,
            x='Nombre',
            y='Commune',
            orientation='h',
            title="🏘️ Top 5 Communes (Hot Spots)",
            color='Nombre',
            color_continuous_scale='Reds'
        )
        fig_communes.update_layout(showlegend=False)
        st.plotly_chart(fig_communes, use_container_width=True)
    
    with col2:
        # Carte de chaleur par intercommunalité
        inter_stats = df_filtered.groupby('Intercommunalité').agg({
            'ID': 'count',
            'Menages': 'sum',
            'Caravanes': 'sum'
        }).reset_index()
        inter_stats.columns = ['Intercommunalité', 'Signalements', 'Ménages', 'Caravanes']
        
        fig_inter_heat = px.bar(
            inter_stats,
            x='Intercommunalité',
            y='Signalements',
            color='Ménages',
            title="🏛️ Activité par Intercommunalité",
            color_continuous_scale='Blues'
        )
        fig_inter_heat.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_inter_heat, use_container_width=True)
    
    # Taille moyenne des groupes par territoire
    st.markdown("#### 👥 Taille Moyenne des Groupes par Territoire")
    
    taille_territoire = df_filtered.groupby('Arrondissement').agg({
        'Menages': 'mean',
        'Caravanes': 'mean',
        'ID': 'count'
    }).reset_index()
    taille_territoire.columns = ['Arrondissement', 'Ménages (moy)', 'Caravanes (moy)', 'Nb signalements']
    
    fig_taille = go.Figure()
    fig_taille.add_trace(go.Bar(
        x=taille_territoire['Arrondissement'],
        y=taille_territoire['Ménages (moy)'],
        name='Ménages',
        marker_color='#3498db'
    ))
    fig_taille.add_trace(go.Bar(
        x=taille_territoire['Arrondissement'],
        y=taille_territoire['Caravanes (moy)'],
        name='Caravanes',
        marker_color='#9b59b6'
    ))
    fig_taille.update_layout(barmode='group')
    st.plotly_chart(fig_taille, use_container_width=True)
    
    # === JURIDIQUE & PROCÉDURES ===
    st.markdown("---")
    st.markdown("### ⚖️ Analyse Juridique")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Répartition juridique
        statut_juridique = df_filtered['Statut_Stationnement'].value_counts().reset_index()
        statut_juridique.columns = ['Statut', 'Nombre']
        
        fig_juridique = px.pie(
            statut_juridique,
            values='Nombre',
            names='Statut',
            title="Répartition par Statut Juridique",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_juridique, use_container_width=True)
    
    with col2:
        # Durée par type de procédure
        duree_proc = df_filtered[df_filtered['Duree_Stationnement'].notna()].groupby('Statut_Stationnement')['Duree_Stationnement'].mean().reset_index()
        duree_proc.columns = ['Procédure', 'Durée moyenne (j)']
        
        fig_duree_proc = px.bar(
            duree_proc,
            x='Procédure',
            y='Durée moyenne (j)',
            title="Durée Moyenne par Type de Procédure",
            color='Durée moyenne (j)',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_duree_proc, use_container_width=True)
    
    with col3:
        # Situation subie vs choisie
        if 'Situation' in df_filtered.columns and not df_filtered['Situation'].isna().all():
            situation = df_filtered['Situation'].value_counts().reset_index()
            situation.columns = ['Situation', 'Nombre']
            
            fig_situation = px.pie(
                situation,
                values='Nombre',
                names='Situation',
                title="Situation : Subie vs Choisie",
                color_discrete_sequence=['#e74c3c', '#27ae60']
            )
            st.plotly_chart(fig_situation, use_container_width=True)
    
    # === SERVICES & CONDITIONS ===
    st.markdown("---")
    st.markdown("### 💧 Services & Conditions de Vie")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Index de précarité
        sans_services = len(df_filtered[
            (df_filtered['Eau'] == 'Non') &
            (df_filtered['Electricite'] == 'Non') &
            (df_filtered['Assainissement'] == 'Non')
        ])
        
        taux_precarite = (sans_services / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        
        st.metric("🚨 Situations de précarité", sans_services,
                 delta=f"{taux_precarite:.0f}% sans aucun service",
                 delta_color="inverse")
        
        # Taux d'équipement
        services_data = {
            'Service': ['Eau', 'Électricité', 'Assainissement'],
            'Disponible': [
                (df_filtered['Eau'] == 'Oui').sum(),
                (df_filtered['Electricite'] == 'Oui').sum(),
                (df_filtered['Assainissement'] == 'Oui').sum()
            ],
            'Non disponible': [
                (df_filtered['Eau'] == 'Non').sum(),
                (df_filtered['Electricite'] == 'Non').sum(),
                (df_filtered['Assainissement'] == 'Non').sum()
            ]
        }
        df_services = pd.DataFrame(services_data)
        
        fig_services = go.Figure()
        fig_services.add_trace(go.Bar(
            x=df_services['Service'],
            y=df_services['Disponible'],
            name='Disponible',
            marker_color='#27ae60'
        ))
        fig_services.add_trace(go.Bar(
            x=df_services['Service'],
            y=df_services['Non disponible'],
            name='Non disponible',
            marker_color='#e74c3c'
        ))
        fig_services.update_layout(
            barmode='stack',
            title="Accès aux Services de Base"
        )
        st.plotly_chart(fig_services, use_container_width=True)
    
    with col2:
        # Corrélation services / durée
        if not df_filtered['Duree_Stationnement'].isna().all():
            df_services_duree = df_filtered[df_filtered['Duree_Stationnement'].notna()].copy()
            df_services_duree['Nb_Services'] = (
                (df_services_duree['Eau'] == 'Oui').astype(int) +
                (df_services_duree['Electricite'] == 'Oui').astype(int) +
                (df_services_duree['Assainissement'] == 'Oui').astype(int)
            )
            
            duree_services = df_services_duree.groupby('Nb_Services')['Duree_Stationnement'].mean().reset_index()
            duree_services.columns = ['Nombre de services', 'Durée moyenne (j)']
            
            fig_corr_services = px.line(
                duree_services,
                x='Nombre de services',
                y='Durée moyenne (j)',
                markers=True,
                title="Impact des Services sur la Durée de Présence",
                line_shape='spline'
            )
            fig_corr_services.update_traces(line_color='#3498db', line_width=3)
            st.plotly_chart(fig_corr_services, use_container_width=True)
    
    # === ANALYSE DES ACTEURS ===
    st.markdown("---")
    st.markdown("### 👥 Mobilisation des Acteurs")
    
    # Analyse des acteurs mobilisés
    all_acteurs = []
    for acteurs in df_filtered['Acteurs'].dropna():
        if acteurs:
            all_acteurs.extend([a.strip() for a in acteurs.split(',')])
    
    if all_acteurs:
        acteurs_counts = pd.Series(all_acteurs).value_counts().head(10).reset_index()
        acteurs_counts.columns = ['Acteur', 'Fréquence']
        
        fig_acteurs = px.bar(
            acteurs_counts,
            x='Fréquence',
            y='Acteur',
            orientation='h',
            title="🤝 Acteurs les Plus Mobilisés",
            color='Fréquence',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_acteurs, use_container_width=True)
    
    # Performance par gestionnaire
    st.markdown("#### 📊 Performance par Gestionnaire")
    
    perf_gestionnaire = df_filtered.groupby('Gestionnaire').agg({
        'ID': 'count',
        'Delai_1ere_Intervention': 'mean',
        'Etat_Gestion': lambda x: (x == 'Fin du stationnement').sum()
    }).reset_index()
    perf_gestionnaire.columns = ['Gestionnaire', 'Nb dossiers', 'Délai moyen (j)', 'Nb terminés']
    perf_gestionnaire['Taux résolution (%)'] = (perf_gestionnaire['Nb terminés'] / perf_gestionnaire['Nb dossiers'] * 100).round(1)
    perf_gestionnaire = perf_gestionnaire.sort_values('Nb dossiers', ascending=False)
    
    st.dataframe(perf_gestionnaire, use_container_width=True, height=300)
    
    # === ANALYSE DU JOURNAL DES INTERVENTIONS ===
    st.markdown("---")
    st.markdown("### 📝 Analyse du Journal des Interventions")
    
    # Extraction et traitement du journal des interventions
    all_interventions = []
    for _, row in df_filtered.iterrows():
        # Le champ 'Journal_Interventions' contient déjà une liste de types d'interventions
        journal = row.get('Journal_Interventions', [])
        if journal:
            # Si c'est une chaîne, la convertir en liste
            if isinstance(journal, str):
                journal = [journal]
            # S'assurer que c'est bien une liste
            elif not isinstance(journal, list):
                continue
                
            for intervention in journal:
                if intervention:  # Ignorer les valeurs vides
                    all_interventions.append({
                        'ID_Dossier': row['ID'],
                        'Commune': row['Commune'],
                        'Intervention': intervention,
                        'Gestionnaire': row['Gestionnaire'],
                        'Etat_Gestion': row['Etat_Gestion'],
                        'Intercommunalité': row['Intercommunalité'],
                        'Arrondissement': row['Arrondissement']
                    })
    
    if all_interventions:
        df_interventions = pd.DataFrame(all_interventions)
        
        # KPIs Interventions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_interventions = len(df_interventions)
            st.metric("📋 Total d'interventions", total_interventions)
        
        with col2:
            dossiers_avec_interventions = df_interventions['ID_Dossier'].nunique()
            st.metric("📁 Dossiers avec interventions", dossiers_avec_interventions)
        
        with col3:
            moy_interventions = total_interventions / dossiers_avec_interventions if dossiers_avec_interventions > 0 else 0
            st.metric("📊 Moyenne par dossier", f"{moy_interventions:.1f}")
        
        with col4:
            dossiers_actifs = len(df_interventions[df_interventions['Etat_Gestion'].str.contains('en cours', case=False, na=False)]['ID_Dossier'].unique())
            st.metric("🔄 Dossiers actifs", dossiers_actifs)
        
        # Graphiques d'analyse
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 des types d'interventions
            st.markdown("#### 🏆 Types d'interventions les plus fréquentes")
            
            interventions_counts = df_interventions['Intervention'].value_counts().head(10).reset_index()
            interventions_counts.columns = ['Type d\'intervention', 'Fréquence']
            
            fig_top_interventions = px.bar(
                interventions_counts,
                y='Type d\'intervention',
                x='Fréquence',
                orientation='h',
                title="Top 10 des Interventions",
                color='Fréquence',
                color_continuous_scale='Blues',
                text='Fréquence'
            )
            fig_top_interventions.update_traces(textposition='outside')
            fig_top_interventions.update_layout(
                showlegend=False,
                height=500,
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_top_interventions, use_container_width=True)
        
        with col2:
            # Répartition par catégorie d'intervention
            st.markdown("#### 📊 Répartition des interventions")
            
            # Créer des catégories d'interventions basées sur les vrais types Airtable
            def categorize_intervention(intervention):
                """Catégorise les interventions selon les types définis dans Airtable"""
                if intervention in ['Appel Téléphonique', 'Contact avec la commune']:
                    return '📞 Contact & Communication'
                elif intervention in ['Visite sur le site', 'Rencontre avec les familles', 'Médiation sociale']:
                    return '🤝 Rencontre & Médiation'
                elif intervention in ['Rédaction PV / plainte', 'Courrier préfecture', 'Demande d\'évacuation']:
                    return '⚖️ Juridique & Administratif'
                elif intervention == 'Intervention police':
                    return '🚔 Forces de l\'Ordre'
                else:
                    return '📋 Autre'
            
            df_interventions['Categorie'] = df_interventions['Intervention'].apply(categorize_intervention)
            
            categories_counts = df_interventions['Categorie'].value_counts().reset_index()
            categories_counts.columns = ['Catégorie', 'Nombre']
            
            fig_categories = px.pie(
                categories_counts,
                values='Nombre',
                names='Catégorie',
                title="Interventions par Catégorie",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_categories.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_categories, use_container_width=True)
        
        # Analyse par gestionnaire
        st.markdown("#### 👥 Performance des Gestionnaires")
        
        gestionnaire_interventions = df_interventions.groupby('Gestionnaire').agg({
            'Intervention': 'count',
            'ID_Dossier': 'nunique'
        }).reset_index()
        gestionnaire_interventions.columns = ['Gestionnaire', 'Nb interventions', 'Nb dossiers']
        gestionnaire_interventions['Moy. interventions/dossier'] = (
            gestionnaire_interventions['Nb interventions'] / gestionnaire_interventions['Nb dossiers']
        ).round(1)
        gestionnaire_interventions = gestionnaire_interventions.sort_values('Nb interventions', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_gestionnaires = px.bar(
                gestionnaire_interventions.head(10),
                x='Gestionnaire',
                y='Nb interventions',
                title="Nombre d'interventions par Gestionnaire (Top 10)",
                color='Moy. interventions/dossier',
                color_continuous_scale='Greens',
                text='Nb interventions'
            )
            fig_gestionnaires.update_traces(textposition='outside')
            fig_gestionnaires.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_gestionnaires, use_container_width=True)
        
        with col2:
            st.dataframe(
                gestionnaire_interventions,
                use_container_width=True,
                height=400
            )
        
        # Analyse territoriale des interventions
        st.markdown("#### 🗺️ Analyse Territoriale des Interventions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interventions par commune
            commune_interventions = df_interventions.groupby('Commune').size().sort_values(ascending=False).head(10).reset_index()
            commune_interventions.columns = ['Commune', 'Nb interventions']
            
            fig_communes_interv = px.bar(
                commune_interventions,
                x='Nb interventions',
                y='Commune',
                orientation='h',
                title="Communes nécessitant le plus d'interventions (Top 10)",
                color='Nb interventions',
                color_continuous_scale='Reds'
            )
            fig_communes_interv.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_communes_interv, use_container_width=True)
        
        with col2:
            # Intensité des interventions (rapport interventions/dossiers par commune)
            commune_intensity = df_interventions.groupby('Commune').agg({
                'Intervention': 'count',
                'ID_Dossier': 'nunique'
            }).reset_index()
            commune_intensity.columns = ['Commune', 'Interventions', 'Dossiers']
            commune_intensity['Intensité'] = (commune_intensity['Interventions'] / commune_intensity['Dossiers']).round(1)
            commune_intensity = commune_intensity.sort_values('Intensité', ascending=False).head(10)
            
            fig_intensity = px.bar(
                commune_intensity,
                x='Intensité',
                y='Commune',
                orientation='h',
                title="Intensité des interventions par commune (Top 10)",
                color='Intensité',
                color_continuous_scale='Oranges',
                hover_data=['Dossiers', 'Interventions']
            )
            fig_intensity.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_intensity, use_container_width=True)
        
        # Tableau détaillé des interventions
        with st.expander("📋 Voir le détail de toutes les interventions"):
            # Options de filtre
            col1, col2 = st.columns(2)
            
            with col1:
                selected_gestionnaire = st.selectbox(
                    "Filtrer par gestionnaire",
                    ['Tous'] + sorted(df_interventions['Gestionnaire'].dropna().unique().tolist())
                )
            
            with col2:
                selected_categorie = st.selectbox(
                    "Filtrer par catégorie",
                    ['Toutes'] + sorted(df_interventions['Categorie'].unique().tolist())
                )
            
            # Application des filtres
            df_interv_filtered = df_interventions.copy()
            
            if selected_gestionnaire != 'Tous':
                df_interv_filtered = df_interv_filtered[df_interv_filtered['Gestionnaire'] == selected_gestionnaire]
            
            if selected_categorie != 'Toutes':
                df_interv_filtered = df_interv_filtered[df_interv_filtered['Categorie'] == selected_categorie]
            
            # Affichage du tableau
            df_interv_display = df_interv_filtered[['ID_Dossier', 'Commune', 'Intervention', 'Categorie', 'Gestionnaire']].copy()
            df_interv_display.columns = ['ID Dossier', 'Commune', 'Type d\'intervention', 'Catégorie', 'Gestionnaire']
            
            st.dataframe(df_interv_display, use_container_width=True, height=400)
            
            st.caption(f"📊 Affichage de {len(df_interv_filtered)} intervention(s)")
            
            # Export des interventions
            csv_interventions = df_interv_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les interventions (CSV)",
                data=csv_interventions,
                file_name=f'nomadia_interventions_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        
        # Insights et recommandations
        st.markdown("#### 💡 Insights & Recommandations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="alert-box alert-warning">', unsafe_allow_html=True)
            top_intervention = interventions_counts.iloc[0]
            st.markdown(f"""
            **Intervention la plus courante:**  
            📌 {top_intervention['Type d\'intervention']}  
            ➡️ {top_intervention['Fréquence']} occurrences ({top_intervention['Fréquence']/total_interventions*100:.1f}%)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="alert-box alert-warning">', unsafe_allow_html=True)
            categorie_dominante = categories_counts.iloc[0]
            st.markdown(f"""
            **Catégorie dominante:**  
            🎯 {categorie_dominante['Catégorie']}  
            ➡️ {categorie_dominante['Nombre']/total_interventions*100:.1f}% des interventions
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="alert-box alert-warning">', unsafe_allow_html=True)
            dossiers_intensifs = len(df_filtered[df_filtered['Nb_Interventions'] > 5])
            st.markdown(f"""
            **Dossiers complexes:**  
            ⚠️ {dossiers_intensifs} dossiers nécessitent  
            plus de 5 interventions
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyse complémentaire : Efficacité des interventions
        st.markdown("#### 🎯 Efficacité des Types d'Interventions")
        
        # Pour chaque type d'intervention, calculer le taux de résolution
        if 'Etat_Gestion' in df_interventions.columns:
            efficacite_interventions = df_interventions.groupby('Intervention').agg({
                'ID_Dossier': lambda x: x.nunique(),
                'Etat_Gestion': lambda x: (x == 'Fin du stationnement').sum()
            }).reset_index()
            efficacite_interventions.columns = ['Type d\'intervention', 'Dossiers', 'Dossiers terminés']
            efficacite_interventions['Taux de résolution (%)'] = (
                efficacite_interventions['Dossiers terminés'] / efficacite_interventions['Dossiers'] * 100
            ).round(1)
            efficacite_interventions = efficacite_interventions.sort_values('Taux de résolution (%)', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_efficacite = px.bar(
                    efficacite_interventions,
                    x='Type d\'intervention',
                    y='Taux de résolution (%)',
                    title="Taux de résolution par type d'intervention",
                    color='Taux de résolution (%)',
                    color_continuous_scale='RdYlGn',
                    text='Taux de résolution (%)'
                )
                fig_efficacite.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_efficacite.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_efficacite, use_container_width=True)
            
            with col2:
                st.dataframe(
                    efficacite_interventions.style.background_gradient(
                        subset=['Taux de résolution (%)'], 
                        cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    height=400
                )
    
    else:
        st.info("ℹ️ Aucune intervention enregistrée dans le journal pour la période sélectionnée.")
    
    # === ANALYSE DE PRÉSENCE (remplace "Stock") ===
    st.markdown("---")
    st.markdown("### 📊 Évolution de la Présence des Voyageurs")
    
    if not df_filtered['Date_Debut'].isna().all():
        # Préparation des données de flux
        flux_data = []
        
        # Nouvelles installations
        for _, row in df_filtered.iterrows():
            if pd.notna(row['Date_Debut']):
                flux_data.append({
                    'date': row['Date_Debut'],
                    'type': 'Installation',
                    'menages': row['Menages'],
                    'caravanes': row['Caravanes']
                })
        
        # Départs
        for _, row in df_filtered.iterrows():
            if pd.notna(row['Date_Fin']):
                flux_data.append({
                    'date': row['Date_Fin'],
                    'type': 'Départ',
                    'menages': -row['Menages'],
                    'caravanes': -row['Caravanes']
                })
        
        if flux_data:
            df_flux = pd.DataFrame(flux_data)
            df_flux['date'] = pd.to_datetime(df_flux['date'])
            df_flux['semaine'] = df_flux['date'].dt.to_period('W').astype(str)
            
            # Agrégation par semaine
            flux_hebdo = df_flux.groupby(['semaine', 'type']).agg({
                'menages': 'sum',
                'caravanes': 'sum'
            }).reset_index()
            
            # Créer un tableau complet avec toutes les semaines
            all_weeks = pd.date_range(
                start=df_flux['date'].min(),
                end=df_flux['date'].max() + pd.Timedelta(days=7),
                freq='W'
            ).to_period('W').astype(str)
            
            # Calcul de la présence cumulée par semaine
            presence_data = []
            present_menages = 0
            present_caravanes = 0
            
            for semaine in sorted(all_weeks):
                installations_menages = flux_hebdo[
                    (flux_hebdo['semaine'] == semaine) & 
                    (flux_hebdo['type'] == 'Installation')
                ]['menages'].sum()
                
                departs_menages = abs(flux_hebdo[
                    (flux_hebdo['semaine'] == semaine) & 
                    (flux_hebdo['type'] == 'Départ')
                ]['menages'].sum())
                
                installations_caravanes = flux_hebdo[
                    (flux_hebdo['semaine'] == semaine) & 
                    (flux_hebdo['type'] == 'Installation')
                ]['caravanes'].sum()
                
                departs_caravanes = abs(flux_hebdo[
                    (flux_hebdo['semaine'] == semaine) & 
                    (flux_hebdo['type'] == 'Départ')
                ]['caravanes'].sum())
                
                present_menages += installations_menages - departs_menages
                present_caravanes += installations_caravanes - departs_caravanes
                
                presence_data.append({
                    'semaine': semaine,
                    'installations_menages': installations_menages,
                    'departs_menages': departs_menages,
                    'present_menages': max(0, present_menages),
                    'installations_caravanes': installations_caravanes,
                    'departs_caravanes': departs_caravanes,
                    'present_caravanes': max(0, present_caravanes)
                })
            
            df_presence = pd.DataFrame(presence_data)
            
            # KPIs de flux
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_installations = df_presence['installations_menages'].sum()
                st.metric("📥 Nouvelles arrivées (ménages)", int(total_installations))
            
            with col2:
                total_departs = df_presence['departs_menages'].sum()
                st.metric("📤 Départs (ménages)", int(total_departs))
            
            with col3:
                presence_actuelle = df_presence['present_menages'].iloc[-1] if len(df_presence) > 0 else 0
                st.metric("👥 Ménages actuellement présents", int(presence_actuelle))
            
            with col4:
                taux_rotation = (total_departs / total_installations * 100) if total_installations > 0 else 0
                st.metric("🔄 Taux de Rotation", f"{taux_rotation:.0f}%")
            
            # Graphique principal : Présence de ménages par semaine
            st.markdown("#### 👥 Évolution de la Présence - Ménages")
            
            fig_presence_menages = go.Figure()
            
            # Ligne de présence
            fig_presence_menages.add_trace(go.Scatter(
                x=df_presence['semaine'],
                y=df_presence['present_menages'],
                mode='lines+markers',
                name='Ménages présents',
                line=dict(color='#e74c3c', width=3),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)',
                hovertemplate='<b>Semaine %{x}</b><br>Présents: %{y} ménages<extra></extra>'
            ))
            
            # Barres d'installations
            fig_presence_menages.add_trace(go.Bar(
                x=df_presence['semaine'],
                y=df_presence['installations_menages'],
                name='Arrivées',
                marker_color='#27ae60',
                opacity=0.6,
                hovertemplate='<b>Semaine %{x}</b><br>Nouvelles arrivées: %{y} ménages<extra></extra>'
            ))
            
            # Barres de départs
            fig_presence_menages.add_trace(go.Bar(
                x=df_presence['semaine'],
                y=df_presence['departs_menages'],
                name='Départs',
                marker_color='#3498db',
                opacity=0.6,
                hovertemplate='<b>Semaine %{x}</b><br>Départs: %{y} ménages<extra></extra>'
            ))
            
            fig_presence_menages.update_layout(
                xaxis_title="Semaine",
                yaxis_title="Nombre de ménages",
                hovermode='x unified',
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            
            st.plotly_chart(fig_presence_menages, use_container_width=True)
            
            # Graphique : Présence de caravanes par semaine
            st.markdown("#### 🚐 Évolution de la Présence - Caravanes")
            
            fig_presence_caravanes = go.Figure()
            
            # Ligne de présence
            fig_presence_caravanes.add_trace(go.Scatter(
                x=df_presence['semaine'],
                y=df_presence['present_caravanes'],
                mode='lines+markers',
                name='Caravanes présentes',
                line=dict(color='#9b59b6', width=3),
                fill='tozeroy',
                fillcolor='rgba(155, 89, 182, 0.2)',
                hovertemplate='<b>Semaine %{x}</b><br>Présentes: %{y} caravanes<extra></extra>'
            ))
            
            # Barres d'installations
            fig_presence_caravanes.add_trace(go.Bar(
                x=df_presence['semaine'],
                y=df_presence['installations_caravanes'],
                name='Arrivées',
                marker_color='#27ae60',
                opacity=0.6,
                hovertemplate='<b>Semaine %{x}</b><br>Nouvelles arrivées: %{y} caravanes<extra></extra>'
            ))
            
            # Barres de départs
            fig_presence_caravanes.add_trace(go.Bar(
                x=df_presence['semaine'],
                y=df_presence['departs_caravanes'],
                name='Départs',
                marker_color='#3498db',
                opacity=0.6,
                hovertemplate='<b>Semaine %{x}</b><br>Départs: %{y} caravanes<extra></extra>'
            ))
            
            fig_presence_caravanes.update_layout(
                xaxis_title="Semaine",
                yaxis_title="Nombre de caravanes",
                hovermode='x unified',
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            
            st.plotly_chart(fig_presence_caravanes, use_container_width=True)
            
            # Tableau de détail hebdomadaire
            with st.expander("📋 Voir le détail hebdomadaire"):
                df_display_presence = df_presence.copy()
                df_display_presence.columns = [
                    'Semaine', 
                    'Arrivées (ménages)', 'Départs (ménages)', 'Présents (ménages)',
                    'Arrivées (caravanes)', 'Départs (caravanes)', 'Présentes (caravanes)'
                ]
                st.dataframe(df_display_presence, use_container_width=True, height=300)
                
                # Export du tableau de flux
                csv_flux = df_display_presence.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger l'analyse de présence (CSV)",
                    data=csv_flux,
                    file_name=f'nomadia_presence_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
    
    # === SAISONNALITÉ ===
    st.markdown("---")
    st.markdown("### 📅 Analyse de Saisonnalité")
    
    if not df_filtered['Date_Debut'].isna().all():
        col1, col2 = st.columns(2)
        
        with col1:
            # Par mois
            df_mois = df_filtered.copy()
            df_mois['Mois'] = df_mois['Date_Debut'].dt.strftime('%Y-%m')
            mois_counts = df_mois.groupby('Mois').agg({
                'ID': 'count',
                'Menages': 'sum'
            }).reset_index()
            mois_counts.columns = ['Mois', 'Signalements', 'Ménages']
            
            fig_mois = go.Figure()
            fig_mois.add_trace(go.Bar(
                x=mois_counts['Mois'],
                y=mois_counts['Signalements'],
                name='Signalements',
                marker_color='#3498db'
            ))
            fig_mois.add_trace(go.Scatter(
                x=mois_counts['Mois'],
                y=mois_counts['Ménages'],
                name='Ménages',
                mode='lines+markers',
                yaxis='y2',
                line=dict(color='#e74c3c', width=3)
            ))
            fig_mois.update_layout(
                title="Évolution Mensuelle",
                yaxis=dict(title='Nombre de signalements'),
                yaxis2=dict(title='Nombre de ménages', overlaying='y', side='right'),
                hovermode='x unified'
            )
            st.plotly_chart(fig_mois, use_container_width=True)
        
        with col2:
            # Par trimestre
            df_trim = df_filtered.copy()
            df_trim['Trimestre'] = df_trim['Date_Debut'].dt.to_period('Q').astype(str)
            trim_counts = df_trim.groupby('Trimestre').size().reset_index(name='Signalements')
            
            fig_trim = px.bar(
                trim_counts,
                x='Trimestre',
                y='Signalements',
                title="Activité par Trimestre",
                color='Signalements',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig_trim, use_container_width=True)
    
    # === TABLEAU DÉTAILLÉ ===
    st.markdown("---")
    st.markdown("### 📋 Liste Détaillée des Signalements")
    
    # Recherche
    search = st.text_input("🔍 Rechercher (commune, adresse, gestionnaire...)", "")
    
    if search:
        df_display = df_filtered[
            df_filtered.apply(lambda row: search.lower() in str(row).lower(), axis=1)
        ]
    else:
        df_display = df_filtered
    
    # Tri par priorité
    sort_option = st.selectbox(
        "Trier par",
        ["Score de priorité (décroissant)", "ID", "Date début", "Délai d'intervention", "Nombre de ménages"]
    )
    
    if sort_option == "Score de priorité (décroissant)":
        df_display = df_display.sort_values('Score_Priorite', ascending=False)
    elif sort_option == "ID":
        df_display = df_display.sort_values('ID')
    elif sort_option == "Date début":
        df_display = df_display.sort_values('Date_Debut', ascending=False)
    elif sort_option == "Délai d'intervention":
        df_display = df_display.sort_values('Delai_1ere_Intervention', ascending=False)
    elif sort_option == "Nombre de ménages":
        df_display = df_display.sort_values('Menages', ascending=False)
    
    # Sélection des colonnes à afficher
    columns_to_display = [
        'ID', 'Commune', 'Intercommunalité', 'Date_Debut', 'Menages', 
        'Caravanes', 'Etat_Gestion', 'Delai_1ere_Intervention', 'Terrain', 'Score_Priorite'
    ]
    
    df_display_table = df_display[columns_to_display].copy()
    
    # Formatage
    if 'Date_Debut' in df_display_table.columns:
        df_display_table['Date_Debut'] = df_display_table['Date_Debut'].dt.strftime('%d/%m/%Y')
    
    df_display_table['Score_Priorite'] = df_display_table['Score_Priorite'].round(1)
    
    # Coloration conditionnelle
    def color_delai(val):
        if pd.isna(val):
            return ''
        elif val <= 7:
            return 'background-color: #d4edda'
        elif val <= 20:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    def color_priorite(val):
        if pd.isna(val):
            return ''
        elif val > 100:
            return 'background-color: #f8d7da; font-weight: bold'
        elif val > 50:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #d4edda'
    
    styled_df = df_display_table.style.applymap(
        color_delai, 
        subset=['Delai_1ere_Intervention']
    ).applymap(
        color_priorite,
        subset=['Score_Priorite']
    )
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Statistiques du tableau
    st.caption(f"📊 Affichage de {len(df_display)} signalement(s) sur {len(df)} au total")
    
    # Export
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name=f'nomadia_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><b>Dashboard Nomadia</b> - Gestion et suivi des situations de stationnement</p>
            <p style='font-size: 0.8rem;'>Données synchronisées avec Airtable via API • Dernière actualisation: {}</p>
            <p style='font-size: 0.7rem; color: #999;'>Ce dashboard est la propriété de la société immatriculée au SIRET : 99158319600019</p>
        </div>
    """.format(datetime.now().strftime("%d/%m/%Y à %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()