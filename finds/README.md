## February 2025 Database Update

CRSP_DATE = 20241231

0. credentials, etc
   - db, user, CRSP_DATE, data path
1. database
   - sql 
     - run create databases
2. structured
   - busday
     - holidays from NYSE website
     - run calendar
   - benchmarks
     - crsp index: sp500.txt.gz, treasuries.txt.gz
     - run sp500, treasuries, French library
   - crsp
     - delist, dist, names, shares, 
     - monthly (DLSTCD, DLSTRET, PRC, RET, RETX)
     - daily (SHROUT, BIDLO, ASKHI, PRC, VOL, RET, BID, ASK, OPENPRC, RETX)
     - treasuries: Index / Treasury and Inflation | US Treasury and Inflation Indexes
   - ibes
     - actpsum, adjsum, idsum, statsum, surpsum (FY1-2, QTR1-4, SEMI1-2, LTG)
   - pstat
     - CRSP | Annual Update | CRSP/Compustat Merged | Compustat CRSP Link: Link and Identifying Info
     - annual: 175 items (6 identifying, 1 company, 93 (92?) balance, 49 income, 17 flow, 8 misc, 2 supp)
     - quarterly: 28 items (4 identifying, 5 company, 18 data, 1 suppl)
     - keydev: 21 items (exclude Headline, duplicated GVKEY and RoleType)
     - supplychain: Linking Queries by WRDS | Supply Chain with IDs (Compustat Segment)
   - cboe
     - indexes

3. readers
   - sectoring
   - fomcreader
   - bea

4. unstructured
   - fomc
     - run fomcreader main
   - edgar
     - Loughran-Mcdonald: https://sraf.nd.edu/sec-edgar-data/cleaned-10x-files/
     - or run edgar._save_10X()
     - run edgar._extract_items()
     - zip -r 2024.zip 2024
     - zip -r mda10K.zip mda10K
     - zip -r bus10K.zip bus10K
     - zip -r qqr10K.zip qqr10K

annual - 175 items
```
GVKEY
CUSIP
CIK
FYR
NAICS
SIC
FYEAR
ACO
ACOX
ACT
AO
AOX
AP
AT
CAPS
CEQ
CEQL
CEQT
CH
CHE
DC
DCLO
DCPSTK
DCVSR
DCVSUB
DCVT
DD
DD1
DD2
DD3
DD4
DD5
DLC
DLTO
DLTP
DLTT
DM
DN
DPACT
DPVIEB
DS
FATB
FATL
GDWL
ICAPT
INTAN
INVFG
INVRM
INVT
INVWIP
ITCB
IVAEQ
IVAO
IVST
LCO
LCOX
LCT
LIFR
LO
LSE
LT
MIB
MRC1
MRC2
MRC3
MRC4
MRC5
MRCT
MSA
NP
OB
PPEGT
PPENT
PPEVEB
PSTK
PSTKC
PSTKL
PSTKN
PSTKR
PSTKRV
REA
REAJO
RECCO
RECD
RECT
RECTA
RECTR
REUNA
SEQ
TLCF
TSTK
TSTKC
TXDB
TXDITC
TXP
TXR
WCAP
XACC
XPP
AQI
AQS
COGS
DO
DP
DVC
DVP
DVT
EBIT
EBITDA
EPSFX
EPSPX
ESUB
FCA
GP
GWO
IB
IBADJ
IBCOM
IDIT
INTC
ITCI
MII
NI
NIADJ
NOPI
NOPIO
OIADP
OIBDP
PI
REVT
SALE
SPI
TXC
TXDI
TXFED
TXFO
TXS
TXT
TXW
XAD
XIDO
XINT
XLR
XOPR
XPR
XRD
XRENT
XSGA
AQC
CAPX
CAPXV
CHECH
DLTIS
DPC
DV
ESUBC
FOPO
IBC
IVCH
OANCF
PRSTKC
SCSTKC
SPPE
SSTK
XIDOC
CSHFD
CSHO
CSHRC
EMP
LIFRP
MIBT
TSTKN
XRDP
PRCC_F
SICH
```

quarterly - 28 items
```
CUSIP
CIK
NAICS
SIC
DATACQTR
DATAFQTR
FQTR
FYEARQ
RDQ
ACTQ
ATQ
CEQQ
CHEQ
COGSQ
CSHOQ
DLCQ
IBQ
LCTQ
LTQ
PPENTQ
PSTKQ
PSTKRQ
REVTQ
SALEQ
SEQQ
TXTQ
XSGAQ
PRCCQ
```
