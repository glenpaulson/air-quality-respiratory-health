"""
City list 

generated using build_cities.py using API calls
  - Coordinates & population: Open-Meteo Geocoding API
  - County & FIPS: U.S. Census Bureau Geographies API
The 50 most populous cities in the United States.
County names use the CDC PLACES convention for the health merge.
"""

CITIES = [
    ('New York'        , 'NY', 'New York'          ,   40.7143,     -74.006),  # FIPS 36061, pop 8804190
    ('Los Angeles'     , 'CA', 'Los Angeles'       ,   34.0522,   -118.2437),  # FIPS 06037, pop 3820914
    ('Chicago'         , 'IL', 'Cook'              ,     41.85,      -87.65),  # FIPS 17031, pop 2664452
    ('Houston'         , 'TX', 'Harris'            ,   29.7633,    -95.3633),  # FIPS 48201, pop 2314157
    ('Phoenix'         , 'AZ', 'Maricopa'          ,   33.4484,    -112.074),  # FIPS 04013, pop 1650070
    ('Philadelphia'    , 'PA', 'Philadelphia'      ,   39.9524,    -75.1636),  # FIPS 42101, pop 1573916
    ('San Antonio'     , 'TX', 'Bexar'             ,   29.4241,    -98.4936),  # FIPS 48029, pop 1526656
    ('San Diego'       , 'CA', 'San Diego'         ,   32.7157,   -117.1647),  # FIPS 06073, pop 1404452
    ('Dallas'          , 'TX', 'Dallas'            ,   32.7831,    -96.8067),  # FIPS 48113, pop 1326087
    ('San Jose'        , 'CA', 'Santa Clara'       ,   37.3394,    -121.895),  # FIPS 06085, pop 997368
    ('Austin'          , 'TX', 'Travis'            ,   30.2672,    -97.7431),  # FIPS 48453, pop 974447
    ('Jacksonville'    , 'FL', 'Duval'             ,   30.3322,    -81.6556),  # FIPS 12031, pop 1009833
    ('Fort Worth'      , 'TX', 'Tarrant'           ,   32.7254,    -97.3208),  # FIPS 48439, pop 1008106
    ('Columbus'        , 'OH', 'Franklin'          ,   39.9612,    -82.9988),  # FIPS 39049, pop 913175
    ('Charlotte'       , 'NC', 'Mecklenburg'       ,   35.2271,    -80.8431),  # FIPS 37119, pop 911311
    ('Indianapolis'    , 'IN', 'Marion'            ,   39.7684,     -86.158),  # FIPS 18097, pop 887642
    ('San Francisco'   , 'CA', 'San Francisco'     ,   37.7749,   -122.4194),  # FIPS 06075, pop 827526
    ('Seattle'         , 'WA', 'King'              ,   47.6062,   -122.3321),  # FIPS 53033, pop 780995
    ('Denver'          , 'CO', 'Denver'            ,   39.7392,   -104.9847),  # FIPS 08031, pop 729019
    ('Washington'      , 'DC', 'District of Columbia',   38.8951,    -77.0364),  # FIPS 11001, pop 689545
    ('Nashville'       , 'TN', 'Davidson'          ,   36.1659,    -86.7844),  # FIPS 47037, pop 689447
    ('Oklahoma City'   , 'OK', 'Oklahoma'          ,   35.4676,    -97.5164),  # FIPS 40109, pop 681054
    ('El Paso'         , 'TX', 'El Paso'           ,   31.7587,   -106.4869),  # FIPS 48141, pop 678815
    ('Boston'          , 'MA', 'Suffolk'           ,   42.3584,    -71.0598),  # FIPS 25025, pop 653833
    ('Portland'        , 'OR', 'Multnomah'         ,   45.5234,   -122.6762),  # FIPS 41051, pop 652503
    ('Las Vegas'       , 'NV', 'Clark'             ,    36.175,   -115.1372),  # FIPS 32003, pop 641903
    ('Detroit'         , 'MI', 'Wayne'             ,   42.3314,    -83.0457),  # FIPS 26163, pop 645705
    ('Memphis'         , 'TN', 'Shelby'            ,   35.1495,     -90.049),  # FIPS 47157, pop 633104
    ('Louisville'      , 'KY', 'Jefferson'         ,   38.2542,    -85.7594),  # FIPS 21111, pop 624444
    ('Baltimore'       , 'MD', 'Baltimore'         ,   39.2904,    -76.6122),  # FIPS 24510, pop 585708
    ('Milwaukee'       , 'WI', 'Milwaukee'         ,   43.0389,    -87.9065),  # FIPS 55079, pop 563531
    ('Albuquerque'     , 'NM', 'Bernalillo'        ,   35.0845,   -106.6511),  # FIPS 35001, pop 564559
    ('Tucson'          , 'AZ', 'Pima'              ,   32.2217,   -110.9265),  # FIPS 04019, pop 542629
    ('Fresno'          , 'CA', 'Fresno'            ,   36.7477,   -119.7724),  # FIPS 06019, pop 542107
    ('Sacramento'      , 'CA', 'Sacramento'        ,   38.5816,   -121.4944),  # FIPS 06067, pop 524943
    ('Kansas City'     , 'MO', 'Jackson'           ,   39.0997,    -94.5786),  # FIPS 29095, pop 475378
    ('Atlanta'         , 'GA', 'Fulton'            ,    33.749,     -84.388),  # FIPS 13121, pop 510823
    ('Miami'           , 'FL', 'Miami-Dade'        ,   25.7743,    -80.1937),  # FIPS 12086, pop 487014
    ('Raleigh'         , 'NC', 'Wake'              ,   35.7721,    -78.6386),  # FIPS 37183, pop 482295
    ('Omaha'           , 'NE', 'Douglas'           ,   41.2563,    -95.9404),  # FIPS 31055, pop 486051
    ('Minneapolis'     , 'MN', 'Hennepin'          ,     44.98,    -93.2638),  # FIPS 27053, pop 410939
    ('Tulsa'           , 'OK', 'Tulsa'             ,    36.154,    -95.9928),  # FIPS 40143, pop 413066
    ('Cleveland'       , 'OH', 'Cuyahoga'          ,   41.4995,    -81.6954),  # FIPS 39035, pop 365379
    ('Wichita'         , 'KS', 'Sedgwick'          ,   37.6922,    -97.3375),  # FIPS 20173, pop 396119
    ('New Orleans'     , 'LA', 'Orleans'           ,   29.9547,    -90.0751),  # FIPS 22071, pop 362701
    ('Salt Lake City'  , 'UT', 'Salt Lake'         ,   40.7608,   -111.8911),  # FIPS 49035, pop 215548
    ('St. Louis'       , 'MO', 'St. Louis'         ,   38.6273,    -90.1979),  # FIPS 29510, pop 279695
    ('Pittsburgh'      , 'PA', 'Allegheny'         ,   40.4406,    -79.9959),  # FIPS 42003, pop 304391
    ('Cincinnati'      , 'OH', 'Hamilton'          ,   39.1271,    -84.5144),  # FIPS 39061, pop 311097
    ('Riverside'       , 'CA', 'Riverside'         ,   33.9534,   -117.3962),  # FIPS 06065, pop 317261
]
