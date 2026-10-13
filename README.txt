CDG QUANT SUITE – STREAMLIT v2.0
================================

CONTENUTO
---------
1) CDG Sector Rotation
2) CDG Stock Score
3) Pagina Metodologia

AVVIO SU WINDOWS
----------------
1. Estrai tutta la cartella dello ZIP.
2. Fai doppio clic su: Avvia_CDG_Quant_Streamlit.bat
3. Al primo avvio viene creata una cartella locale .venv e vengono installate le librerie.
4. L'app si apre nel browser.

È necessario avere Python 3.11/3.12 installato.

============================================================
1. CDG SECTOR ROTATION
============================================================

INPUT CONSIGLIATO
-----------------
- 3 anni di dati GIORNALIERI.
- 10 ETF/indici settoriali + MSCI World.
- Preferire prezzi Adjusted / Total Return coerenti tra tutte le serie.

SCORE DI DEFAULT
----------------
Composite Score:
- Momentum: 45%
- Relative Strength vs MSCI World: 25%
- Trend: 20%
- Risk-adjusted Momentum: 10%

Momentum:
- 3M: 30%
- 6M: 40%
- 12M: 20%
- 12-1: 10%

OUTPUT OPERATIVO
----------------
- Top 5 settori.
- Download dei prezzi giornalieri dei Top 5 sugli ultimi 12 mesi.

PORTFOLIO OPTIMIZER
-------------------
Configurazione operativa stabilita:
- Universo: Top 5 CDG Sector Rotation
- Lookback: 1 anno DAILY
- Metodo: Mean-Variance + Black-Litterman
- Peso minimo: 0%
- Peso massimo: 40%
- Ribilanciamento: mensile

============================================================
2. CDG STOCK SCORE
============================================================

OBIETTIVO
---------
Creare una fase di selezione quantitativa dei singoli titoli PRIMA di Markowitz.

Flusso:
Ufficio Studi
→ titoli appartenenti ai Top 5 settori CDG
→ CDG Stock Score
→ candidati finali
→ Markowitz / Black-Litterman

IMPORTANTE
----------
Il peso del settore nel portafoglio ETF NON viene trasferito ai singoli titoli.
Lo Stock Score usa il settore come benchmark di forza relativa, non come peso da replicare.

INPUT CONSIGLIATO
-----------------
- 3 anni daily per i titoli.
- Gli stessi ETF/indici settoriali usati nel CDG Sector Rotation.
- La pagina Stock Score può riutilizzare automaticamente i dati e i Top 5 già calcolati nella pagina Sector Rotation.

MAPPATURA
---------
Per ogni titolo devi indicare il benchmark settoriale corretto tramite la tabella modificabile nell'app.
Esempio:
Microsoft → Information Technology
JPMorgan → Financials
Exxon → Energy

Dalla v3.3 il benchmark viene PRECOMPILATO dal market data incorporato
(market_data_settori.csv, ricavato da MARKET_DATA_AZIONI Q3 2026):
- abbinamento per ISIN, poi nome (ignorando PLC/AG/SpA/Inc...), poi nome simile;
- settori Morningstar convertiti in GICS (Technology → Information Technology,
  Consumer Cyclical → Consumer Discretionary, Consumer Defensive → Consumer Staples,
  Financial Services → Financials, Healthcare → Health Care, Basic Materials → Materials);
- colonna "Abbinamento": Trovato / Da verificare (nome simile) /
  Benchmark settore non caricato / Non trovato;
- le correzioni manuali hanno sempre la priorità e restano anche cambiando modulo;
- lo Stock Score non parte finché ci sono titoli senza benchmark.
Un market data aggiornato (Excel/CSV con colonne Nome, ISIN, Settore) si può
caricare dal riquadro "Market data settori" della sezione 3.

STOCK SCORE DI DEFAULT
----------------------
A) RELATIVE STRENGTH VS PROPRIO SETTORE – 35%
   - RS 3M: 30%
   - RS 6M: 45%
   - RS 12M: 25%
   50 = in linea con il proprio settore.
   >50 = sovraperformance.
   <50 = sottoperformance.

B) MOMENTUM – 25%
   - 3M: 30%
   - 6M: 40%
   - 12-1: 30%
   È momentum assoluto del titolo.

C) TREND – 20%
   - distanza da MA100: 30%
   - distanza da MA200: 40%
   - pendenza MA200: 30%

D) RISK QUALITY – 20%
   - downside volatility 126 giorni: 35%
   - max drawdown 126 giorni: 30%
   - tracking error vs settore 126 giorni: 35%
   Un titolo con rischio idiosincratico più contenuto riceve un punteggio più elevato.

SELEZIONE DI DEFAULT
--------------------
- Top 2 titoli per ciascun settore eleggibile.
- Massimo 10 titoli complessivi.
- Parametri modificabili nell'interfaccia.

MARKOWITZ SUI TITOLI
--------------------
L'app esporta automaticamente 12 mesi daily dei candidati selezionati.
Questi dati sono pensati per il tuo Portfolio Optimizer.

Lo Stock Score NON determina i pesi finali.

AUDIT INTEGRATO
---------------
La pagina Stock Score calcola:
- correlazione media fra i quattro blocchi dello score;
- Information Coefficient (Spearman) tra score e rendimento del mese successivo;
- stabilità della selezione perturbando i pesi dello score di ±20%;
- backtest diagnostico Equal Weight dei titoli selezionati vs tutti i titoli caricati.

ATTENZIONE SUL BACKTEST
-----------------------
Il backtest diagnostico dello Stock Score NON sostituisce il backtest reale della strategia dell'Ufficio Studi.
Per un backtest rigorosamente point-in-time serve la lista storica dei titoli che erano effettivamente selezionati dall'Ufficio Studi a ogni data mensile.
Usare oggi la lista corrente per il passato introdurrebbe survivorship/look-ahead bias.

============================================================
PROCEDURA MENSILE CONSIGLIATA
============================================================

1. Scarica/aggiorna 3 anni daily dei 10 settori + MSCI World.
2. Esegui CDG Sector Rotation.
3. Registra i Top 5 settori.
4. Prendi la lista corrente dei titoli selezionati dall'Ufficio Studi.
5. Carica 3 anni daily dei titoli nel CDG Stock Score.
6. Associa ogni titolo al proprio benchmark settoriale.
7. Mantieni eleggibili i Top 5 settori CDG.
8. Esegui CDG Stock Score.
9. Controlla ranking, audit e candidati finali.
10. Scarica i 12 mesi daily dei candidati.
11. Caricali nel Portfolio Optimizer.
12. Esegui Mean-Variance + Black-Litterman e crea il basket del mese.
13. Ribilancia mensilmente.

FILE PRINCIPALI
---------------
streamlit_app.py
sector_rotation_engine.py
stock_score_engine.py
requirements.txt
Avvia_CDG_Quant_Streamlit.bat

============================================================
FIX v2.2 – AVVIO WINDOWS
============================================================
Corretto il pacchetto di installazione:
- aggiunto requirements.txt;
- il BAT verifica se le librerie sono davvero installate;
- se un precedente avvio ha lasciato una .venv incompleta, il BAT la ripara;
- se requirements.txt non fosse disponibile, il BAT dispone anche di una
  installazione di emergenza delle dipendenze.

Se provieni dalla v2.1 e vuoi evitare qualunque residuo:
1. estrai la v2.2 in una NUOVA cartella;
2. avvia Avvia_CDG_Quant_Streamlit.bat.


============================================================
v3.0 – CDG FACTOR LEADERSHIP MONITOR
============================================================

Nuovo terzo modulo della suite:

LeaderScore =
40% Rank rendimento 6M
+ 30% Rank rendimento 3M
+ 20% Rank rendimento 12M
+ 10% Relative Strength vs MSCI World

NOTA IMPORTANTE SULLA RELATIVE STRENGTH
---------------------------------------
Sottrarre MSCI World al rendimento di tutti gli ETF e poi fare un semplice
ranking darebbe praticamente lo stesso ordine del momentum. Per evitare
questa duplicazione, la componente RS è assoluta:

- 50 = performance in linea con MSCI World
- > 50 = sovraperformance
- < 50 = sottoperformance

Default: RS 6 mesi, con scala 10%.

USO OPERATIVO
-------------
Il Factor Leadership Monitor NON assegna i pesi.

Serve a:
1. ordinare gli ETF/fattori per leadership;
2. confrontare il ranking con i pesi del Markowitz;
3. misurare la correlazione LeaderScore / peso;
4. misurare il peso allocato ai Top 3 leader;
5. controllare storico, persistenza e IC diagnostico.

Input consigliato:
- 3 anni daily per avere uno storico di audit significativo;
- almeno ~13 mesi daily per il solo score corrente;
- ETF Smart Beta + MSCI World nello stesso formato già supportato dalla suite.

Portfolio Optimizer:
- continua a utilizzare 1 anno daily;
- Mean-Variance + Black-Litterman;
- 0–40%;
- ribilanciamento mensile.

CSV:
I download della suite usano la virgola come separatore decimale.


============================================================
FIX v3.1 – NOMI ETF SMART BETA / MSCI WORLD
============================================================
Corretto un errore di aliasing dell'importatore.

Prima:
iShares Edge MSCI World Momentum...
iShares Edge MSCI World Value...
Xtrackers MSCI World Quality...
Xtrackers MSCI World Minimum Volatility...

potevano essere erroneamente rinominati:
MSCI World (2), MSCI World (3), MSCI World (4), ecc.

Ora:
- i nomi degli ETF Smart Beta vengono preservati;
- soltanto il broad MSCI World standard viene riconosciuto come "MSCI World";
- Momentum, Value, Quality, Min Vol, Multifactor, Equal Weight, Dividend,
  Size ecc. non possono più essere scambiati automaticamente per il benchmark;
- nel Factor Leadership Monitor compare un controllo esplicito dell'universo
  prima del calcolo.


============================================================
v3.2 – FIX AUDIT + SHORTLIST FACTOR LEADERS
============================================================
- Eliminato l'errore matplotlib nello Storico Leadership.
- Shortlist Top N LeaderScore, default Top 4.
- Export 12M daily della shortlist per Portfolio Optimizer.
- Audit migliorato LeaderScore vs pesi Markowitz.
- Un leader a peso zero non è automaticamente un errore:
  Markowitz ottimizza anche volatilità, correlazioni e rendimenti impliciti.

Test consigliato:
A) Markowitz su tutto l'universo Smart Beta.
B) LeaderScore -> Top 4/5 -> Markowitz 1Y daily 0-40%.
Confrontare soprattutto Sharpe, drawdown, turnover e stabilita'.
