ecv_to_udb/
│
├── convert_ecv_to_udb.py          ← entry point
├── requirements.txt               
├── README.md                      
│
├── src/
│   ├── __init__.py
│   ├── constants.py               ← maps, recode tables, column lists
│   ├── readers.py                 ← raw ECV file readers
│   ├── recode.py                  ← all recoding logic
│   ├── household.py               ← household-level UDB builder
│   ├── person.py                  ← person-level UDB builder
│   ├── merge.py                   ← merge, validate, export
│   └── schemas.py                 ← pandera UDB output schemas
│
├── input_data/
|    |---euromod_data                    
│
└── output/
    ├── ES_2017_a2.txt
    ├── ES_2018_a1.txt
    └── ES_2019_b1.txt