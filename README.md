# Regen: Generate Regular Expressions from Words

Regen is a Python library for creating [regular expressions](https://en.wikipedia.org/wiki/Regular_expression) from a finite set of words, and vice versa. Utilizing constraint programming (CP) powered by [Google OR-Tools](https://developers.google.com/optimization), it finds the *near-shortest* regular expression that matches precisely the same words as the input.

- Expands regular expressions into a list of words that they can match.
- Generates an optimal regular expression from a given list of words.

Currently, regen supports only "finite matching," meaning it can generate patterns that match a specific, finite set of strings. For instance, it can produce `[AB]C?` (matching `A`, `B`, `AC`, `BC`), but not `[AB]C*`, which would match an unlimited number of `C`s. And it can only expand the former type of patterns.

## Examples

### Two-Letter Abbreviations of the 50 US States

`AL, AK, AZ, AR, CA, CZ, CO, CT, DE, DC, FL, GA, GU, HI, ID, IL, IN, IA, KS, KY, LA, ME, MD, MA, MI, MN, MS, MO, MT, NE, NV, NH, NJ, NM, NY, NC, ND, OH, OK, OR, PA, PR, RI, SC, SD, TN, TX, UT, VT, VI, VA, WA, WV, WI, WY`

**Result:**

`A[KLRZ]|C[AOTZ]|DC|DE|FL|GU|HI|I[ADLN]|KS|KY|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|PR|RI|SC|SD|TN|TX|UT|V[AIT]|W[AIVY]|[GLP]A`

### Two and Three Letters ISO Country Codes (498 Codes of 249 Countries)

```
AD, AND, AE, ARE, AF, AFG, AG, ATG, AI, AIA, AL, ALB, AM, ARM, AO, AGO, AQ, ATA, AR, ARG, AS, ASM, AT, AUT, AU, AUS, AW, ABW, AX, ALA, AZ, AZE, BA, BIH, BB, BRB, BD, BGD, BE, BEL, BF, BFA, BG, BGR, BH, BHR, BI, BDI, BJ, BEN, BL, BLM, BM, BMU, BN, BRN, BO, BOL, BQ, BES, BR, BRA, BS, BHS, BT, BTN, BV, BVT, BW, BWA, BY, BLR, BZ, BLZ, CA, CAN, CC, CCK, CD, COD, CF, CAF, CG, COG, CH, CHE, CI, CIV, CK, COK, CL, CHL, CM, CMR, CN, CHN, CO, COL, CR, CRI, CU, CUB, CV, CPV, CW, CUW, CX, CXR, CY, CYP, CZ, CZE, DE, DEU, DJ, DJI, DK, DNK, DM, DMA, DO, DOM, DZ, DZA, EC, ECU, EE, EST, EG, EGY, EH, ESH, ER, ERI, ES, ESP, ET, ETH, FI, FIN, FJ, FJI, FK, FLK, FM, FSM, FO, FRO, FR, FRA, GA, GAB, GB, GBR, GD, GRD, GE, GEO, GF, GUF, GG, GGY, GH, GHA, GI, GIB, GL, GRL, GM, GMB, GN, GIN, GP, GLP, GQ, GNQ, GR, GRC, GS, SGS, GT, GTM, GU, GUM, GW, GNB, GY, GUY, HK, HKG, HM, HMD, HN, HND, HR, HRV, HT, HTI, HU, HUN, ID, IDN, IE, IRL, IL, ISR, IM, IMN, IN, IND, IO, IOT, IQ, IRQ, IR, IRN, IS, ISL, IT, ITA, JE, JEY, JM, JAM, JO, JOR, JP, JPN, KE, KEN, KG, KGZ, KH, KHM, KI, KIR, KM, COM, KN, KNA, KP, PRK, KR, KOR, KW, KWT, KY, CYM, KZ, KAZ, LA, LAO, LB, LBN, LC, LCA, LI, LIE, LK, LKA, LR, LBR, LS, LSO, LT, LTU, LU, LUX, LV, LVA, LY, LBY, MA, MAR, MC, MCO, MD, MDA, ME, MNE, MF, MAF, MG, MDG, MH, MHL, MK, MKD, ML, MLI, MM, MMR, MN, MNG, MO, MAC, MP, MNP, MQ, MTQ, MR, MRT, MS, MSR, MT, MLT, MU, MUS, MV, MDV, MW, MWI, MX, MEX, MY, MYS, MZ, MOZ, NA, NAM, NC, NCL, NE, NER, NF, NFK, NG, NGA, NI, NIC, NL, NLD, NO, NOR, NP, NPL, NR, NRU, NU, NIU, NZ, NZL, OM, OMN, PA, PAN, PE, PER, PF, PYF, PG, PNG, PH, PHL, PK, PAK, PL, POL, PM, SPM, PN, PCN, PR, PRI, PS, PSE, PT, PRT, PW, PLW, PY, PRY, QA, QAT, RE, REU, RO, ROU, RS, SRB, RU, RUS, RW, RWA, SA, SAU, SB, SLB, SC, SYC, SD, SDN, SE, SWE, SG, SGP, SH, SHN, SI, SVN, SJ, SJM, SK, SVK, SL, SLE, SM, SMR, SN, SEN, SO, SOM, SR, SUR, SS, SSD, ST, STP, SV, SLV, SX, SXM, SY, SYR, SZ, SWZ, TC, TCA, TD, TCD, TF, ATF, TG, TGO, TH, THA, TJ, TJK, TK, TKL, TL, TLS, TM, TKM, TN, TUN, TO, TON, TR, TUR, TT, TTO, TV, TUV, TW, TWN, TZ, TZA, UA, UKR, UG, UGA, UM, UMI, US, USA, UY, URY, UZ, UZB, VA, VAT, VC, VCT, VE, VEN, VG, VGB, VI, VIR, VN, VNM, VU, VUT, WF, WLF, WS, WSM, YE, YEM, YT, MYT, ZA, ZAF, ZM, ZMB, ZW, ZWE
```

**Result:**

```
A(B?W|G?O|LB|N?D|R?[EM]|SM?|T?F|U[ST]?|ZE?|[FRT]?G|[ILT]A?|[QRX])|B(DI?|E[LNS]?|G[DR]?|H[RS]?|IH?|L[MRZ]?|MU?|OL?|R[ABN]?|TN?|VT?|[ABJNQSYZ]|[FW]A?)|C(A?F|AN?|CK?|H[ELN]?|IV?|O[DGKLM]?|P?V|RI?|U?W|UB?|Y[MP]?|ZE?|[DGKLN]|[MX]R?)|D(EU?|JI?|N?K|OM?|[MZ]A?)|E(CU?|E|GY?|H|RI?|S[HPT]?|TH?)|F(IN?|JI?|L?K|R?O|RA?|S?M)|G(B?R|EO?|GY?|HA?|I?N|L?P|N?Q|R?[DL]|RC|TM?|U[FMY]?|[AFIMSWY]|[AIMN]?B)|H(KG?|RV?|TI?|UN?|[MN]D?)|I(ND?|OT?|R[LNQ]?|S[LR]?|TA?|[DM]N?|[ELQ])|J(A?M|EY?|OR?|PN?)|K(A?Z|EN?|GZ?|H?M|IR?|NA?|O?R|WT?|[HPY])|L(B?[RY]|BN?|IE?|TU?|UX?|[AS]O?|[CKV]A?)|M(A?R|A?[CF]|C?O|D|D?[AV]|E?X|HL?|KD?|N|N?[EP]|O?Z|T?Q|US?|YS?|[DN]?G|[LRY]?T|[LW]I?|[MS]R?)|N(AM?|FK?|GA?|I[CU]?|LD?|RU?|U|[CPZ]L?|[EO]R?)|OMN?|P(AK?|ER?|HL?|L?W|N?G|O?L|R[IKTY]?|SE?|Y?F|[AC]?N|[KMTY])|QAT?|R(U|U?S|WA?|[EO]U?)|S(AU?|G?S|L[BEV]?|P?M|R?B|S?D|V?K|W?[EZ]|Y?C|[DEHV]?N|[GT]P?|[HIVY]|[JOX]M?|[MUY]?R)|T(C?D|F|JK?|K|K?[LM]|LS|U?[NRV]|[CHZ]A?|[GT]O?|[OW]N?)|U(A|KR|MI?|R?Y|ZB?|[GS]A?)|V(EN?|GB?|IR?|NM?|[ACU]T?)|W(L?F|SM?)|Y(EM?|T)|Z(AF?|MB?|WE?)
```

## Usage

### Example 1

Convert a list of words into a regular expression:

```python
>>> from regen import Regen

>>> wordlist = ['ABC', 'ABD', 'BBC', 'BBD']
>>> regen = Regen(wordlist)
>>> result = regen.to_regex()
>>> result
'[AB]B[CD]'
```

### Example 2

Expand regular expressions into a word list:

```python
>>> from regen import Regen

>>> wordlist = ['[AB]B[CD]', 'XYZ']
>>> regen = Regen(wordlist)
>>> words = regen.to_words()
>>> words
['ABC', 'ABD', 'BBC', 'BBD', 'XYZ']
```

Then convert it into a new regular expression:

```python
>>> result = regen.to_regex()
>>> result
'(XYZ|[AB]B[CD])'

>>> result = regen.to_regex(omitOuterParen=True)
>>> result
'XYZ|[AB]B[CD]'
```

### Example 3

regen.py can be used as a command-line utility:

```bash
regen.py --compute "cat" "bat" "at" "fat|boat"
# Output: (bo?|c|f)?at

regen.py --extract "[AB]C[DE]"
# Output:
# ACD
# ACE
# BCD
# BCE

regen.py -f words.txt
# Compute the regex from a word list file, with one word per line.
```

## Author

- **David Pi**
