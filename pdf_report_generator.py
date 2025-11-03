# ==========================================
# pdf_report_generator.py (McKINSEY LEVEL 10/10)
# Synthèse Mensuelle Premium - Design Exécutif
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import io
from typing import Tuple, Dict, List
import warnings
import traceback
import sys
import linecache

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION DESIGN MCKINSEY
# ==========================================

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['axes.edgecolor'] = '#d1d5db'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.color'] = '#f3f4f6'
plt.rcParams['grid.linewidth'] = 0.5

# Palette professionnelle McKinsey
COLORS = {
    'navy': '#003d7a',
    'success': '#00a86b',
    'alert': '#e63946',
    'warning': '#f4a261',
    'light_bg': '#f9fafb',
    'dark_text': '#111827',
    'medium_text': '#4b5563',
    'light_text': '#9ca3af',
    'border': '#e5e7eb',
    'accent_blue': '#457b9d',
}


class NomadiaMonthlyReport:
    """
    Générateur de rapports mensuels McKinsey-level.
    Structure: 7 pages rigoureuses et professionnelles.
    """
    
    def __init__(self, df: pd.DataFrame, month: int, year: int):
        self.df = df.copy()
        self.month = month
        self.year = year
        self.month_name = datetime(year, month, 1).strftime('%B %Y')
        self.month_short = datetime(year, month, 1).strftime('%b %Y')
        self.df_month = self._filter_month_data()
        self.page_count = 7
        self.current_page = 0
    
    def _filter_month_data(self) -> pd.DataFrame:
        df = self.df.copy()
        df['Date_Debut'] = pd.to_datetime(df['Date_Debut'], errors='coerce')
        mask = (df['Date_Debut'].dt.year == self.year) & (df['Date_Debut'].dt.month == self.month)
        return df[mask]
    
    def generate_pdf(self) -> io.BytesIO:
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            self._add_page(pdf, self._page_1_cover)
            self._add_page(pdf, self._page_2_executive_summary)
            self._add_page(pdf, self._page_3_territorial)
            self._add_page(pdf, self._page_4_presence_trends)
            self._add_page(pdf, self._page_5_operational)
            self._add_page(pdf, self._page_6_quality)
            self._add_page(pdf, self._page_7_recommendations)
            
            d = pdf.infodict()
            d['Title'] = f'Rapport Mensuel - {self.month_short}'
            d['Author'] = 'Nomadia Direction'
            d['Subject'] = 'Synthèse mensuelle de gestion des signalements'
        
        pdf_buffer.seek(0)
        return pdf_buffer
    
    def _add_page(self, pdf, page_func):
        """Wrapper pour ajouter une page avec numérotation."""
        self.current_page += 1
        fig = page_func()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _add_header_footer(self, fig, title: str, is_cover=False):
        """Ajoute en-tête et pied de page standardisés."""
        if not is_cover:
            # En-tête
            ax_header = fig.add_axes([0, 0.96, 1, 0.04])
            ax_header.set_facecolor('white')
            ax_header.axis('off')
            
            fig.text(0.08, 0.975, 'NOMADIA', fontsize=9, fontweight='bold',
                    color=COLORS['navy'])
            fig.text(0.92, 0.975, f'{self.month_short} | Page {self.current_page}/{self.page_count}',
                    fontsize=8, color=COLORS['light_text'], ha='right')
            
            # Ligne séparatrice
            fig.add_axes([0.08, 0.955, 0.84, 0.005]).set_facecolor(COLORS['navy'])
            fig.add_axes([0.08, 0.955, 0.84, 0.005]).axis('off')
            
            # Titre
            fig.text(0.08, 0.93, title, fontsize=16, fontweight='bold',
                    color=COLORS['navy'])
            
            # Pied de page
            fig.text(0.08, 0.01, f'© 2024 Nomadia | Confidentiel | {datetime.now().strftime("%d.%m.%Y")}',
                    fontsize=7, color=COLORS['light_text'])
    
    def _page_1_cover(self) -> plt.Figure:
        """Page 1 : Couverture Executive Premium."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        # Header bleu marine
        ax_header = fig.add_axes([0, 0.75, 1, 0.25])
        ax_header.set_facecolor(COLORS['navy'])
        ax_header.axis('off')
        
        fig.text(0.5, 0.92, 'NOMADIA', fontsize=24, fontweight='bold',
                color='white', ha='center')
        fig.text(0.5, 0.85, 'RAPPORT DE SYNTHÈSE MENSUELLE', fontsize=14,
                color='white', ha='center', style='italic')
        fig.text(0.5, 0.78, self.month_name.upper(), fontsize=18, fontweight='bold',
                color=COLORS['success'], ha='center')
        
        # KPIs - Design card standardisé
        y_pos = 0.65
        kpis = self._get_kpis()
        
        kpi_items = [
            ('SIGNALEMENTS TRAITÉS', str(kpis['total']), COLORS['navy']),
            ('MÉNAGES SUR LE TERRAIN', str(int(kpis['menages'])), COLORS['accent_blue']),
            ('DÉLAI MOYEN 1ère INTERVENTION', f"{kpis['delai']:.1f} j", 
             COLORS['success'] if kpis['delai'] <= 7 else COLORS['alert']),
            ('TAUX DE RÉACTIVITÉ', f"{kpis['reactivite']:.0f}%",
             COLORS['success'] if kpis['reactivite'] >= 70 else COLORS['alert']),
        ]
        
        for i, item in enumerate(kpi_items):
            label = item[0]
            value = item[1]
            color = item[2]
            y = y_pos - (i * 0.13)
            # Card avec bordure gauche colorée
            rect = plt.Rectangle((0.1, y - 0.09), 0.8, 0.11,
                                transform=fig.transFigure, facecolor=COLORS['light_bg'],
                                edgecolor=COLORS['border'], linewidth=1)
            fig.patches.append(rect)
            
            # Barre gauche colorée
            rect_accent = plt.Rectangle((0.1, y - 0.09), 0.02, 0.11,
                                       transform=fig.transFigure, facecolor=color,
                                       edgecolor='none')
            fig.patches.append(rect_accent)
            
            fig.text(0.12, y - 0.02, label, fontsize=8, fontweight='bold',
                    color=COLORS['dark_text'], transform=fig.transFigure)
            fig.text(0.12, y - 0.065, value, fontsize=20, fontweight='bold',
                    color=color, transform=fig.transFigure)
        
        # Footer info
        footer_text = f"""
Période: {self.month_name} | Dossiers actifs: {kpis['actifs']} | Caravanes: {int(kpis['caravanes'])}
Ratio: {kpis['ratio']:.2f} caravanes/ménage | Généré: {datetime.now().strftime('%d.%m.%Y')}
        """
        fig.text(0.5, 0.15, footer_text, fontsize=8, color=COLORS['medium_text'],
                ha='center', family='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['light_bg'],
                         edgecolor=COLORS['border'], linewidth=1))
        
        return fig
    
    def _page_2_executive_summary(self) -> plt.Figure:
        """Page 2 : Résumé Exécutif."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        self._add_header_footer(fig, 'Résumé Exécutif')
        
        gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3,
                     top=0.92, bottom=0.08, left=0.08, right=0.92)
        
        kpis = self._get_kpis()
        
        # 1. Evolution hebdo
        ax1 = fig.add_subplot(gs[0, :])
        self.df_month['Semaine'] = self.df_month['Date_Debut'].dt.to_period('W')
        weekly = self.df_month.groupby('Semaine').size()
        
        ax1.fill_between(range(len(weekly)), weekly.values, alpha=0.15, color=COLORS['navy'])
        ax1.plot(range(len(weekly)), weekly.values, marker='o', linewidth=2.5,
                color=COLORS['navy'], markersize=8, markerfacecolor='white',
                markeredgewidth=2, markeredgecolor=COLORS['navy'])
        
        ax1.set_xticks(range(len(weekly)))
        ax1.set_xticklabels([str(w) for w in weekly.index], fontsize=8)
        ax1.set_ylabel('Signalements', fontsize=9, fontweight='bold', color=COLORS['dark_text'])
        ax1.set_facecolor(COLORS['light_bg'])
        ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Délai distribution
        ax2 = fig.add_subplot(gs[1, 0])
        delais = self.df_month['Delai_1ere_Intervention'].dropna()
        
        ax2.hist(delais, bins=12, color=COLORS['navy'], alpha=0.7, edgecolor='white', linewidth=1.5)
        ax2.axvline(7, color=COLORS['success'], linestyle='--', linewidth=2, label='Objectif')
        ax2.axvline(20, color=COLORS['warning'], linestyle='--', linewidth=2, label='Seuil')
        ax2.set_xlabel('Jours', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Dossiers', fontsize=9, fontweight='bold')
        ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax2.set_facecolor(COLORS['light_bg'])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. État gestion
        ax3 = fig.add_subplot(gs[1, 1])
        etat = self.df_month['Etat_Gestion'].value_counts().head(4)
        colors_etat = [COLORS['success'], COLORS['navy'], COLORS['warning'], COLORS['alert']]
        
        bars = ax3.barh(range(len(etat)), etat.values, color=colors_etat[:len(etat)],
                       edgecolor='white', linewidth=1.5)
        ax3.set_yticks(range(len(etat)))
        ax3.set_yticklabels(etat.index, fontsize=8)
        ax3.set_xlabel('Dossiers', fontsize=9, fontweight='bold')
        ax3.invert_yaxis()
        ax3.set_facecolor(COLORS['light_bg'])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        for i, (bar, v) in enumerate(zip(bars, etat.values)):
            ax3.text(v + 0.1, i, f'{int(v)}', va='center', fontsize=8, fontweight='bold')
        
        # 4. Synthèse texte
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        summary = f"""
POINTS CLÉS

Réactivité opérationnelle          {kpis['reactivite']:.0f}% d'interventions < 7 jours
Délai moyen d'intervention         {kpis['delai']:.1f} jours (objectif: 7 jours)
Charge active                      {kpis['actifs']} dossiers en cours
Interventions moyennes             {kpis['interv_moy']:.1f} par dossier
Population impactée                {int(kpis['menages'])} ménages | {int(kpis['caravanes'])} caravanes
        """
        
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                color=COLORS['dark_text'],
                bbox=dict(boxstyle='round,pad=1.2', facecolor=COLORS['light_bg'],
                         edgecolor=COLORS['border'], linewidth=1))
        
        return fig
    
    def _page_3_territorial(self) -> plt.Figure:
        """Page 3 : Analyse Territoriale."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        self._add_header_footer(fig, 'Analyse Territoriale')
        
        gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                     top=0.92, bottom=0.08, left=0.08, right=0.92)
        
        # 1. Top communes
        ax1 = fig.add_subplot(gs[0, :])
        top_communes = self.df_month['Commune'].value_counts().head(10)
        colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_communes)))
        
        bars = ax1.barh(range(len(top_communes)), top_communes.values,
                       color=colors_gradient, edgecolor='white', linewidth=1.5)
        ax1.set_yticks(range(len(top_communes)))
        ax1.set_yticklabels(top_communes.index, fontsize=8)
        ax1.set_xlabel('Signalements', fontsize=9, fontweight='bold')
        ax1.invert_yaxis()
        ax1.set_facecolor(COLORS['light_bg'])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        for i, v in enumerate(top_communes.values):
            ax1.text(v + 0.1, i, f'{int(v)}', va='center', fontsize=8, fontweight='bold')
        
        # 2. Intercommunalités
        ax2 = fig.add_subplot(gs[1, 0])
        inter = self.df_month['Intercommunalité'].value_counts()
        colors_inter = plt.cm.Blues(np.linspace(0.5, 0.9, len(inter)))
        
        bars = ax2.bar(range(len(inter)), inter.values, color=colors_inter,
                      edgecolor='white', linewidth=1.5, width=0.6)
        ax2.set_xticks(range(len(inter)))
        ax2.set_xticklabels(inter.index, rotation=30, ha='right', fontsize=8)
        ax2.set_ylabel('Signalements', fontsize=9, fontweight='bold')
        ax2.set_facecolor(COLORS['light_bg'])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for i, (bar, v) in enumerate(zip(bars, inter.values)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(v)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 3. Arrondissements
        ax3 = fig.add_subplot(gs[1, 1])
        arron = self.df_month['Arrondissement'].value_counts()
        colors_arron = plt.cm.Greens(np.linspace(0.5, 0.9, len(arron)))
        
        bars = ax3.bar(range(len(arron)), arron.values, color=colors_arron,
                      edgecolor='white', linewidth=1.5, width=0.6)
        ax3.set_xticks(range(len(arron)))
        ax3.set_xticklabels(arron.index, rotation=30, ha='right', fontsize=8)
        ax3.set_ylabel('Signalements', fontsize=9, fontweight='bold')
        ax3.set_facecolor(COLORS['light_bg'])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        for i, (bar, v) in enumerate(zip(bars, arron.values)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(v)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 4. Accès services
        ax4 = fig.add_subplot(gs[2, :])
        services = ['Eau', 'Électricité', 'Assainissement']
        available = [
            (self.df_month['Eau'] == 'Oui').sum(),
            (self.df_month['Electricite'] == 'Oui').sum(),
            (self.df_month['Assainissement'] == 'Oui').sum()
        ]
        unavailable = [
            (self.df_month['Eau'] == 'Non').sum(),
            (self.df_month['Electricite'] == 'Non').sum(),
            (self.df_month['Assainissement'] == 'Non').sum()
        ]
        
        x = np.arange(len(services))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, available, width, label='Disponible',
                       color=COLORS['success'], edgecolor='white', linewidth=1.5)
        bars2 = ax4.bar(x + width/2, unavailable, width, label='Non disponible',
                       color=COLORS['alert'], edgecolor='white', linewidth=1.5)
        
        ax4.set_ylabel('Dossiers', fontsize=9, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(services, fontsize=9, fontweight='bold')
        ax4.legend(fontsize=8, loc='upper right')
        ax4.set_facecolor(COLORS['light_bg'])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        return fig
    
    def _page_4_presence_trends(self) -> plt.Figure:
        """Page 4 : Tendances de Présence (NOUVELLE PAGE PREMIUM)."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        self._add_header_footer(fig, 'Évolution de la Présence - Tendances')
        
        gs = GridSpec(2, 1, figure=fig, hspace=0.4, top=0.92, bottom=0.08,
                    left=0.08, right=0.92)
        
        # Calculer flux présence
        flux_data = []
        
        for _, row in self.df.iterrows():
            try:
                if pd.notna(row.get('Date_Debut')):
                    flux_data.append({
                        'date': row.get('Date_Debut'),
                        'type': 'Installation',
                        'menages': row.get('Menages', 0),
                        'caravanes': row.get('Caravanes', 0)
                    })
                
                if pd.notna(row.get('Date_Fin')):
                    flux_data.append({
                        'date': row.get('Date_Fin'),
                        'type': 'Départ',
                        'menages': -row.get('Menages', 0),
                        'caravanes': -row.get('Caravanes', 0)
                    })
            except Exception as e:
                continue
        
        if len(flux_data) == 0:
            # Si pas de données, afficher message
            ax1 = fig.add_subplot(gs[0])
            ax1.text(0.5, 0.5, 'Données de présence insuffisantes', 
                    ha='center', va='center', fontsize=12, transform=ax1.transAxes)
            ax1.axis('off')
            return fig
        
        df_flux = pd.DataFrame(flux_data)
        df_flux['date'] = pd.to_datetime(df_flux['date'])
        df_flux['semaine'] = df_flux['date'].dt.to_period('W').astype(str)
        
        # Agrégation hebdomadaire
        flux_hebdo = df_flux.groupby(['semaine', 'type']).agg({
            'menages': 'sum',
            'caravanes': 'sum'
        }).reset_index()
        
        all_weeks = sorted(df_flux['semaine'].unique())
        
        if len(all_weeks) == 0:
            ax1 = fig.add_subplot(gs[0])
            ax1.text(0.5, 0.5, 'Pas de semaines avec données', 
                    ha='center', va='center', fontsize=12, transform=ax1.transAxes)
            ax1.axis('off')
            return fig
        
        # Calcul cumulatif - ✅ ASSIGNATION CORRECTE (PAS DE DÉBALLAGE)
        presence_data = []
        present_menages = 0
        present_caravanes = 0
        
        for semaine in all_weeks:
            # ✅ Assignation simple - pas de déballage triple
            inst_m_result = flux_hebdo[(flux_hebdo['semaine'] == semaine) & 
                                    (flux_hebdo['type'] == 'Installation')]['menages'].sum()
            inst_m = inst_m_result if not pd.isna(inst_m_result) else 0
            
            dep_m_result = flux_hebdo[(flux_hebdo['semaine'] == semaine) & 
                                    (flux_hebdo['type'] == 'Départ')]['menages'].sum()
            dep_m = abs(dep_m_result) if not pd.isna(dep_m_result) else 0
            
            inst_c_result = flux_hebdo[(flux_hebdo['semaine'] == semaine) & 
                                    (flux_hebdo['type'] == 'Installation')]['caravanes'].sum()
            inst_c = inst_c_result if not pd.isna(inst_c_result) else 0
            
            dep_c_result = flux_hebdo[(flux_hebdo['semaine'] == semaine) & 
                                    (flux_hebdo['type'] == 'Départ')]['caravanes'].sum()
            dep_c = abs(dep_c_result) if not pd.isna(dep_c_result) else 0
            
            present_menages += inst_m - dep_m
            present_caravanes += inst_c - dep_c
            
            presence_data.append({
                'semaine': semaine,
                'arrivees_m': inst_m,
                'departs_m': dep_m,
                'present_m': max(0, present_menages),
                'arrivees_c': inst_c,
                'departs_c': dep_c,
                'present_c': max(0, present_caravanes)
            })
        
        df_presence = pd.DataFrame(presence_data)
        
        # 1. Ménages
        ax1 = fig.add_subplot(gs[0])
        
        x_range = range(len(df_presence))
        ax1.fill_between(x_range, df_presence['present_m'].values, 
                        alpha=0.2, color=COLORS['navy'])
        ax1.plot(x_range, df_presence['present_m'].values, 
                marker='o', linewidth=2.5, color=COLORS['navy'], markersize=7,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor=COLORS['navy'],
                label='Présents')
        
        ax1.bar([x - 0.2 for x in x_range], df_presence['arrivees_m'].values, 
            width=0.4, color=COLORS['success'], alpha=0.6, label='Arrivées')
        ax1.bar([x + 0.2 for x in x_range], -df_presence['departs_m'].values,
            width=0.4, color=COLORS['alert'], alpha=0.6, label='Départs')
        
        ax1.set_ylabel('Ménages', fontsize=9, fontweight='bold')
        ax1.set_title('Évolution de la présence - Ménages', fontsize=11, fontweight='bold',
                    loc='left', pad=10)
        ax1.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax1.set_facecolor(COLORS['light_bg'])
        ax1.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Caravanes
        ax2 = fig.add_subplot(gs[1])
        
        ax2.fill_between(x_range, df_presence['present_c'].values,
                        alpha=0.2, color=COLORS['accent_blue'])
        ax2.plot(x_range, df_presence['present_c'].values,
                marker='s', linewidth=2.5, color=COLORS['accent_blue'], markersize=7,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor=COLORS['accent_blue'],
                label='Présentes')
        
        ax2.bar([x - 0.2 for x in x_range], df_presence['arrivees_c'].values,
            width=0.4, color=COLORS['success'], alpha=0.6, label='Arrivées')
        ax2.bar([x + 0.2 for x in x_range], -df_presence['departs_c'].values,
            width=0.4, color=COLORS['alert'], alpha=0.6, label='Départs')
        
        ax2.set_ylabel('Caravanes', fontsize=9, fontweight='bold')
        ax2.set_xlabel('Semaine', fontsize=9, fontweight='bold')
        ax2.set_title('Évolution de la présence - Caravanes', fontsize=11, fontweight='bold',
                    loc='left', pad=10)
        ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax2.set_facecolor(COLORS['light_bg'])
        ax2.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        return fig
    
    # ============================================================
# FIX PAGE 5 - _page_5_operational()
# ============================================================
# REMPLACER LA SECTION PIE CHART (ligne ~535-545)

    def _page_5_operational(self) -> plt.Figure:
        """Page 5 : Performance Opérationnelle."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        self._add_header_footer(fig, 'Performance Opérationnelle')
        
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                    top=0.92, bottom=0.08, left=0.08, right=0.92)
        
        # 1. Qualité intervention
        ax1 = fig.add_subplot(gs[0, 0])
        delais = self.df_month['Delai_1ere_Intervention'].dropna()
        
        if len(delais) > 0:
            rapides = len(delais[delais <= 7])
            lents = len(delais[delais > 20])
            moyens = len(delais) - rapides - lents
            
            colors_react = [COLORS['success'], COLORS['warning'], COLORS['alert']]
            sizes = [rapides, moyens, lents]
            labels = [f'< 7j\n{rapides}\n{rapides/len(delais)*100:.0f}%',
                    f'7-20j\n{moyens}\n{moyens/len(delais)*100:.0f}%',
                    f'> 20j\n{lents}\n{lents/len(delais)*100:.0f}%']
            
            # ✅ FIX: NE PAS DÉPILER LE RÉSULTAT DE pie()
            # pie() retourne (wedges, texts, autotexts) SEULEMENT avec certains paramètres
            # Pour éviter l'erreur, n'assignez pas le résultat
            ax1.pie(sizes, labels=labels, colors=colors_react,
                    textprops={'fontsize': 8, 'fontweight': 'bold'},
                    wedgeprops=dict(edgecolor='white', linewidth=2))
            
            ax1.set_title('Distribution Délai', fontsize=10, fontweight='bold', loc='left', pad=10)
        else:
            ax1.text(0.5, 0.5, 'Pas de données de délai', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.axis('off')
        
        # 2. Interventions
        ax2 = fig.add_subplot(gs[0, 1])
        interv_dist = self.df_month['Nb_Interventions'].value_counts().sort_index().head(6)
        
        if len(interv_dist) > 0:
            bars = ax2.bar(range(len(interv_dist)), interv_dist.values,
                        color=COLORS['navy'], edgecolor='white', linewidth=1.5)
            ax2.set_xticks(range(len(interv_dist)))
            ax2.set_xticklabels(interv_dist.index, fontsize=8)
            ax2.set_xlabel('Nombre d\'interventions', fontsize=8, fontweight='bold')
            ax2.set_ylabel('Dossiers', fontsize=8, fontweight='bold')
            ax2.set_facecolor(COLORS['light_bg'])
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            for i, (bar, v) in enumerate(zip(bars, interv_dist.values)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(v)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Pas de données', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        # 3. État gestion vs durée
        ax3 = fig.add_subplot(gs[1, :])
        valid = self.df_month.dropna(subset=['Nb_Interventions', 'Duree_Stationnement'])
        
        if len(valid) > 0:
            scatter = ax3.scatter(valid['Nb_Interventions'], valid['Duree_Stationnement'],
                                s=valid['Menages']*20 + 50, c=valid['Menages'],
                                cmap='Blues', alpha=0.6, edgecolors=COLORS['navy'],
                                linewidth=1.5)
            
            ax3.set_xlabel('Nombre d\'interventions', fontsize=9, fontweight='bold')
            ax3.set_ylabel('Durée présence (jours)', fontsize=9, fontweight='bold')
            ax3.set_title('Corrélation Interventions vs Durée', fontsize=10, fontweight='bold',
                        loc='left', pad=10)
            ax3.set_facecolor(COLORS['light_bg'])
            ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            
            cbar = plt.colorbar(scatter, ax=ax3, pad=0.02)
            cbar.set_label('Ménages', fontsize=8, fontweight='bold')
            cbar.ax.tick_params(labelsize=7)
        else:
            ax3.text(0.5, 0.5, 'Pas de données de corrélation', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        return fig
    
    def _page_6_quality(self) -> plt.Figure:
        """Page 6 : Qualité & Acteurs."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        self._add_header_footer(fig, 'Qualité & Acteurs Mobilisés')
        
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                     top=0.92, bottom=0.08, left=0.08, right=0.92)
        
        # 1. Types interventions
        ax1 = fig.add_subplot(gs[0, 0])
        all_interv = []
        for _, row in self.df_month.iterrows():
            journal = row.get('Journal_Interventions', [])
            if journal:
                if isinstance(journal, str):
                    journal = [journal]
                all_interv.extend([i for i in journal if i])
        
        if all_interv:
            interv_counts = pd.Series(all_interv).value_counts().head(6)
            colors_interv = plt.cm.Blues(np.linspace(0.5, 0.9, len(interv_counts)))
            bars = ax1.barh(range(len(interv_counts)), interv_counts.values,
                           color=colors_interv, edgecolor='white', linewidth=1.5)
            ax1.set_yticks(range(len(interv_counts)))
            ax1.set_yticklabels(interv_counts.index, fontsize=8)
            ax1.set_xlabel('Fréquence', fontsize=8, fontweight='bold')
            ax1.invert_yaxis()
            ax1.set_facecolor(COLORS['light_bg'])
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            for i, (bar, v) in enumerate(zip(bars, interv_counts.values)):
                ax1.text(v + 0.2, i, f'{int(v)}', va='center', fontsize=8, fontweight='bold')
        
        ax1.set_title('Types d\'interventions', fontsize=10, fontweight='bold', loc='left', pad=10)
        
        # 2. Gestionnaires
        ax2 = fig.add_subplot(gs[0, 1])
        gest = self.df_month['Gestionnaire'].value_counts().head(6)
        colors_gest = plt.cm.Greens(np.linspace(0.5, 0.9, len(gest)))
        bars = ax2.bar(range(len(gest)), gest.values, color=colors_gest,
                      edgecolor='white', linewidth=1.5, width=0.6)
        ax2.set_xticks(range(len(gest)))
        ax2.set_xticklabels(gest.index, rotation=30, ha='right', fontsize=8)
        ax2.set_ylabel('Dossiers', fontsize=8, fontweight='bold')
        ax2.set_facecolor(COLORS['light_bg'])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for i, (bar, v) in enumerate(zip(bars, gest.values)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(v)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax2.set_title('Charge par gestionnaire', fontsize=10, fontweight='bold', loc='left', pad=10)
        
        # 3. Synthèse conditions
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        sans_services = len(self.df_month[
            (self.df_month['Eau'] == 'Non') &
            (self.df_month['Electricite'] == 'Non') &
            (self.df_month['Assainissement'] == 'Non')
        ])
        
        kpis = self._get_kpis()
        
        summary = f"""
CONDITIONS DE VIE ET ENJEUX SOCIAUX

Population en grande précarité      {sans_services} ménages sans aucun service ({sans_services/len(self.df_month)*100:.0f}%)
Accès à l'eau                       {(self.df_month['Eau'] == 'Oui').sum()} / {len(self.df_month)} dossiers ({(self.df_month['Eau'] == 'Oui').sum()/len(self.df_month)*100:.0f}%)
Accès à l'électricité               {(self.df_month['Electricite'] == 'Oui').sum()} / {len(self.df_month)} dossiers ({(self.df_month['Electricite'] == 'Oui').sum()/len(self.df_month)*100:.0f}%)
Accès à l'assainissement            {(self.df_month['Assainissement'] == 'Oui').sum()} / {len(self.df_month)} dossiers ({(self.df_month['Assainissement'] == 'Oui').sum()/len(self.df_month)*100:.0f}%)

Interventions moyennes              {kpis['interv_moy']:.1f} interventions par dossier
Dossiers résolus                    {len(self.df_month[self.df_month['Etat_Gestion'] == 'Fin du stationnement'])} clôturés ({len(self.df_month[self.df_month['Etat_Gestion'] == 'Fin du stationnement'])/len(self.df_month)*100:.0f}%)
        """
        
        ax3.text(0.05, 0.95, summary, transform=ax3.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                color=COLORS['dark_text'],
                bbox=dict(boxstyle='round,pad=1.2', facecolor=COLORS['light_bg'],
                         edgecolor=COLORS['border'], linewidth=1))
        
        return fig
    
    def _page_7_recommendations(self) -> plt.Figure:
        """Page 7 : Recommandations & Conclusion."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        self._add_header_footer(fig, 'Conclusions & Recommandations')
        
        kpis = self._get_kpis()
        strengths = self._get_strengths(kpis)
        improvements = self._get_improvements(kpis)
        actions = self._get_actions(kpis)
        
        y_pos = 0.88
        
        # Synthèse
        fig.text(0.08, y_pos, 'SYNTHÈSE', fontsize=11, fontweight='bold',
                color=COLORS['navy'])
        
        synthesis = f"""
Mois: {self.month_name} | Signalements: {kpis['total']} | Ménages: {int(kpis['menages'])}
Réactivité: {kpis['reactivite']:.0f}% (objectif 70%) | Délai moyen: {kpis['delai']:.1f}j (objectif 7j)
Dossiers actifs: {kpis['actifs']} | Interventions moyennes: {kpis['interv_moy']:.1f}
        """
        
        fig.text(0.08, y_pos - 0.09, synthesis, fontsize=8, color=COLORS['medium_text'],
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=COLORS['light_bg'],
                         edgecolor=COLORS['border'], linewidth=1))
        
        # Points forts
        y_pos -= 0.22
        fig.text(0.08, y_pos, 'POINTS FORTS', fontsize=10, fontweight='bold',
                color=COLORS['success'])
        
        for i, strength in enumerate(strengths[:4]):
            fig.text(0.10, y_pos - 0.045 - (i * 0.035), f"• {strength}",
                    fontsize=8.5, color=COLORS['dark_text'])
        
        # Points d'attention
        y_pos -= 0.22
        fig.text(0.08, y_pos, 'POINTS D\'ATTENTION', fontsize=10, fontweight='bold',
                color=COLORS['alert'])
        
        for i, improvement in enumerate(improvements[:4]):
            fig.text(0.10, y_pos - 0.045 - (i * 0.035), f"• {improvement}",
                    fontsize=8.5, color=COLORS['dark_text'])
        
        # Actions prioritaires
        y_pos -= 0.22
        fig.text(0.08, y_pos, 'ACTIONS PRIORITAIRES', fontsize=10, fontweight='bold',
                color=COLORS['navy'])
        
        for i, action in enumerate(actions[:5]):
            fig.text(0.10, y_pos - 0.045 - (i * 0.035), f"• {action}",
                    fontsize=8.5, color=COLORS['dark_text'])
        
        return fig
    
    def _get_kpis(self) -> Dict:
        """Calcule les KPIs critiques."""
        df = self.df_month
        
        today = pd.Timestamp.now()
        actifs = df[(df['Date_Fin'].isna()) | (df['Date_Fin'] >= today)]
        
        delai_moyen = df['Delai_1ere_Intervention'].mean()
        interv_moyen = df['Nb_Interventions'].mean()
        
        rapides = len(df[df['Delai_1ere_Intervention'] <= 7])
        reactivite = (rapides / len(df) * 100) if len(df) > 0 else 0
        
        menages = df['Menages'].sum()
        caravanes = df['Caravanes'].sum()
        ratio = caravanes / menages if menages > 0 else 0
        
        return {
            'total': len(df),
            'menages': menages,
            'caravanes': caravanes,
            'ratio': ratio,
            'actifs': len(actifs),
            'delai': delai_moyen if not pd.isna(delai_moyen) else 0,
            'reactivite': reactivite,
            'interv_moy': interv_moyen if not pd.isna(interv_moyen) else 0,
        }
    
    def _get_strengths(self, kpis) -> List[str]:
        """Points forts du mois."""
        strengths = []
        
        if kpis['reactivite'] >= 70:
            strengths.append(f"Excellente réactivité: {kpis['reactivite']:.0f}% d'interventions < 7j")
        elif kpis['reactivite'] >= 50:
            strengths.append(f"Bonne réactivité: {kpis['reactivite']:.0f}%")
        
        if kpis['delai'] <= 7:
            strengths.append(f"Objectif délai atteint: {kpis['delai']:.1f}j en moyenne")
        
        resolved = len(self.df_month[self.df_month['Etat_Gestion'] == 'Fin du stationnement'])
        if resolved / len(self.df_month) > 0.3 if len(self.df_month) > 0 else False:
            strengths.append(f"Bonne résolution: {resolved} dossiers clôturés")
        
        strengths.append(f"Activité soutenue: {kpis['total']} signalements gérés ce mois")
        
        return strengths if strengths else ["Suivi régulier maintenu"]
    
    def _get_improvements(self, kpis) -> List[str]:
        """Points à améliorer."""
        improvements = []
        
        if kpis['reactivite'] < 70:
            improvements.append(f"Réactivité insuffisante: {kpis['reactivite']:.0f}% (cible 70%)")
        
        if kpis['delai'] > 7:
            improvements.append(f"Délai trop long: {kpis['delai']:.1f}j (objectif 7j)")
        
        if kpis['actifs'] > 15:
            improvements.append(f"Accumulation: {kpis['actifs']} dossiers en cours")
        
        no_interv = len(self.df_month[self.df_month['Nb_Interventions'] == 0])
        if no_interv > 0:
            improvements.append(f"Dossiers bloqués: {no_interv} sans intervention")
        
        return improvements if improvements else ["Situation maîtrisée"]
    
    def _get_actions(self, kpis) -> List[str]:
        """Actions recommandées."""
        actions = []
        
        if kpis['reactivite'] < 70:
            actions.append("Renforcer les ressources pour améliorer la réactivité")
        
        if kpis['delai'] > 7:
            actions.append("Analyser les freins aux délais d'intervention")
        
        if kpis['actifs'] > 15:
            actions.append("Réunion d'équipe: accélérer les clôtures en cours")
        
        no_interv = len(self.df_month[self.df_month['Nb_Interventions'] == 0])
        if no_interv > 0:
            actions.append(f"Priorité immédiate: traiter les {no_interv} dossiers bloqués")
        
        actions.append("Validation mensuelle avec les gestionnaires territoriaux")
        
        return actions


# ==========================================
# INTÉGRATION STREAMLIT
# ==========================================

def generate_monthly_report_streamlit(df: pd.DataFrame, month: int, year: int) -> io.BytesIO:
    """Génère un rapport mensuel PDF McKinsey-level avec DIAGNOSTIC COMPLET."""
    try:
        report = NomadiaMonthlyReport(df, month, year)
        pdf_buffer = report.generate_pdf()
        return pdf_buffer
    except ValueError as e:
        if "not enough values to unpack" in str(e):
            print("\n" + "🔴"*40)
            print("🎯 ERREUR UNPACKING TROUVÉE!")
            print("🔴"*40)
            
            # Parse l'erreur
            error_msg = str(e)
            print(f"\n❌ Message d'erreur: {error_msg}")
            
            # Affiche le traceback complet
            print("\n📍 TRACEBACK COMPLET:")
            exc_type, exc_value, exc_tb = sys.exc_info()
            
            # Remonte le traceback jusqu'à la ligne fautive
            tb_list = []
            tb = exc_tb
            while tb is not None:
                tb_list.append(tb)
                tb = tb.tb_next
            
            # Affiche tous les appels
            for i, tb in enumerate(tb_list):
                frame = tb.tb_frame
                lineno = tb.tb_lineno
                filename = frame.f_code.co_filename
                funcname = frame.f_code.co_name
                
                # Récupère la ligne de code
                line = linecache.getline(filename, lineno).strip()
                
                print(f"\n  [{i}] {funcname}() - Ligne {lineno}")
                print(f"      Fichier: {filename}")
                print(f"      Code: {line}")
                
                # Affiche les variables locales pertinentes
                if i == len(tb_list) - 1:  # Dernière frame = la coupable
                    print(f"\n      🔍 VARIABLES LOCALES À CETTE LIGNE:")
                    for var_name, var_value in frame.f_locals.items():
                        if not var_name.startswith('_'):
                            try:
                                val_str = str(var_value)[:150]
                                var_type = type(var_value).__name__
                                
                                # Cas spécial: si c'est un tuple/liste
                                if isinstance(var_value, (tuple, list)):
                                    print(f"         {var_name}: {var_type} avec {len(var_value)} éléments = {val_str}")
                                else:
                                    print(f"         {var_name}: {var_type} = {val_str}")
                            except:
                                print(f"         {var_name}: {type(var_value).__name__} (non affichable)")
            
            print("\n" + "🔴"*40)
            print("💡 SOLUTION: Cherchez la ligne avec:")
            print("   - a, b, c = quelquechose")
            print("   - for i, (x, y, z) in enumerate(...)")
            print("   Où 'quelquechose' ne retourne que 2 valeurs!")
            print("🔴"*40 + "\n")
        
        raise Exception(f"Erreur génération PDF: {str(e)}")
    
    except Exception as e:
        print("\n" + "❌"*40)
        print("ERREUR NON-PRÉVUE:")
        print("❌"*40)
        traceback.print_exc()
        print("❌"*40 + "\n")
        raise Exception(f"Erreur génération PDF: {str(e)}")
        
class MultiMonthReportGenerator:
    """Génère des rapports comparatifs multi-mois."""
    
    def __init__(self, df: pd.DataFrame, months_list: List[Tuple[int, int]]):
        self.df = df
        self.months_list = months_list
    
    def generate_comparison_pdf(self) -> io.BytesIO:
        """PDF de comparaison multi-mois."""
        pdf_buffer = io.BytesIO()
        
        with PdfPages(pdf_buffer) as pdf:
            self._page_cover(pdf)
            self._page_comparison(pdf)
        
        pdf_buffer.seek(0)
        return pdf_buffer
    
    def _page_cover(self, pdf):
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        ax_header = fig.add_axes([0, 0.75, 1, 0.25])
        ax_header.set_facecolor(COLORS['navy'])
        ax_header.axis('off')
        
        fig.text(0.5, 0.92, 'NOMADIA', fontsize=24, fontweight='bold',
                color='white', ha='center')
        fig.text(0.5, 0.85, 'RAPPORT COMPARATIF', fontsize=14,
                color='white', ha='center', style='italic')
        
        months_str = ', '.join([f"{m[0]:02d}/{m[1]}" for m in self.months_list])
        fig.text(0.5, 0.60, f'Période: {months_str}', ha='center', fontsize=12,
                color=COLORS['dark_text'])
        fig.text(0.5, 0.50, f'Généré: {datetime.now().strftime("%d.%m.%Y")}',
                ha='center', fontsize=9, color=COLORS['medium_text'])
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _page_comparison(self, pdf):
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                     top=0.92, bottom=0.08, left=0.08, right=0.92)
        
        fig.text(0.08, 0.965, 'Comparaison des performances', fontsize=12, fontweight='bold',
                color=COLORS['navy'])
        
        kpis_data = []
        for month, year in self.months_list:
            report = NomadiaMonthlyReport(self.df, month, year)
            kpis = report._get_kpis()
            kpis['period'] = f"{month:02d}/{year}"
            kpis_data.append(kpis)
        
        df_kpis = pd.DataFrame(kpis_data)
        
        # Signalements
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df_kpis['period'], df_kpis['total'], marker='o', linewidth=2.5,
                color=COLORS['navy'], markersize=8)
        ax1.fill_between(range(len(df_kpis)), df_kpis['total'], alpha=0.15,
                        color=COLORS['navy'])
        ax1.set_title('Signalements', fontsize=9, fontweight='bold', loc='left')
        ax1.set_facecolor(COLORS['light_bg'])
        ax1.grid(True, alpha=0.2)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Réactivité
        ax2 = fig.add_subplot(gs[0, 1])
        colors = [COLORS['success'] if r >= 70 else COLORS['alert']
                 for r in df_kpis['reactivite']]
        ax2.bar(range(len(df_kpis)), df_kpis['reactivite'], color=colors,
               edgecolor='white', linewidth=1.5)
        ax2.axhline(y=70, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_xticks(range(len(df_kpis)))
        ax2.set_xticklabels(df_kpis['period'], fontsize=8)
        ax2.set_title('Réactivité (%)', fontsize=9, fontweight='bold', loc='left')
        ax2.set_facecolor(COLORS['light_bg'])
        ax2.grid(True, alpha=0.2, axis='y')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Délai
        ax3 = fig.add_subplot(gs[1, 0])
        colors = [COLORS['success'] if d <= 7 else COLORS['alert']
                 for d in df_kpis['delai']]
        ax3.bar(range(len(df_kpis)), df_kpis['delai'], color=colors,
               edgecolor='white', linewidth=1.5)
        ax3.axhline(y=7, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax3.set_xticks(range(len(df_kpis)))
        ax3.set_xticklabels(df_kpis['period'], fontsize=8)
        ax3.set_title('Délai moyen (j)', fontsize=9, fontweight='bold', loc='left')
        ax3.set_facecolor(COLORS['light_bg'])
        ax3.grid(True, alpha=0.2, axis='y')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Ménages
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df_kpis['period'], df_kpis['menages'], marker='s', linewidth=2.5,
                color=COLORS['accent_blue'], markersize=8)
        ax4.fill_between(range(len(df_kpis)), df_kpis['menages'], alpha=0.15,
                        color=COLORS['accent_blue'])
        ax4.set_title('Population (ménages)', fontsize=9, fontweight='bold', loc='left')
        ax4.set_facecolor(COLORS['light_bg'])
        ax4.grid(True, alpha=0.2)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close(fig)


# ==========================================
# HELPERS
# ==========================================

def get_available_months(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Liste des mois disponibles."""
    df_temp = df.copy()
    df_temp['Date_Debut'] = pd.to_datetime(df_temp['Date_Debut'], errors='coerce')
    months = df_temp['Date_Debut'].dt.to_period('M').dropna().unique()
    return [(m.month, m.year) for m in sorted(months)]


def validate_month(month: int, year: int, df: pd.DataFrame) -> bool:
    """Vérifie si le mois existe."""
    available = get_available_months(df)
    return (month, year) in available