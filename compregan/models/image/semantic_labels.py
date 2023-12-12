from enum import unique, IntEnum


@unique
class SemanticLabels(IntEnum):
    pass


class CityScapesLabels(SemanticLabels):
    Unlabeled = 0
    EgoVehicle = 1
    RectificationBorder = 2
    OutOfRoi = 3
    Static = 4
    Dynamic = 5
    Ground = 6
    Road = 7
    Sidewalk = 8
    Parking = 9
    RailTrack = 10
    Building = 11
    Wall = 12
    Fence = 13
    GuardRail = 14
    Bridge = 15
    Tunnel = 16
    Pole = 17
    Polegroup = 18
    TrafficLight = 19
    TrafficSign = 20
    Vegetation = 21
    Terrain = 22
    Sky = 23
    Person = 24
    Rider = 25
    Car = 26
    Truck = 27
    Bus = 28
    Caravan = 29
    Trailer = 30
    Train = 31
    Motorcycle = 32
    Bicycle = 33
    LicensePlate = -1


class NYUDepthLabels(SemanticLabels):
    AirConditioner = 79
    AirDuct = 38
    AirVent = 25
    Alarm = 525
    AlarmClock = 156
    Album = 822
    AluminiumFoil = 708
    AmericanFlag = 870
    Antenna = 796
    Apple = 334
    Ashtray = 377
    Avocado = 680
    BabyChair = 494
    BabyGate = 591
    BackScrubber = 656
    Backpack = 206
    Bag = 55
    BagOfBagels = 690
    BagOfChips = 245
    BagOfFlour = 285
    BagOfHotDogBuns = 747
    BagOfOreo = 692
    Bagel = 689
    BakingDish = 260
    Ball = 60
    Balloon = 385
    Banana = 147
    BananaPeel = 691
    Banister = 453
    Bar = 51
    BarOfSoap = 564
    Barrel = 343
    Baseball = 825
    Basket = 39
    Basketball = 542
    BasketballHoop = 162
    Bassinet = 414
    Bathtub = 136
    BeanBag = 797
    Bed = 157
    BedSheets = 352
    BeddingPackage = 808
    Beeper = 780
    Belt = 610
    Bench = 204
    Bicycle = 189
    BicycleHelmet = 337
    Bin = 307
    Binder = 399
    Blackboard = 225
    Blanket = 312
    Blender = 268
    Blinds = 80
    Board = 408
    Book = 1
    BookHolder = 827
    Bookend = 374
    Bookrack = 224
    Books = 85
    Bookshelf = 88
    Boomerang = 773
    Bottle = 2
    BottleOfComet = 755
    BottleOfContactLensSolution = 633
    BottleOfHandWashLiquid = 677
    BottleOfKetchup = 750
    BottleOfLiquid = 685
    BottleOfListerine = 676
    BottleOfPerfume = 840
    BottleOfSoap = 502
    Bowl = 22
    Box = 26
    BoxOfPaper = 503
    BoxOfZiplockBags = 271
    Bracelet = 860
    Bread = 246
    Brick = 695
    Briefcase = 617
    Broom = 328
    Bucket = 427
    Bulb = 688
    BunkBed = 804
    BusinessCards = 535
    ButterflySculpture = 712
    Button = 774
    Cabinet = 3
    CableBox = 168
    CableModem = 73
    CableRack = 104
    Cables = 450
    Cactus = 641
    Cake = 289
    Calculator = 200
    Calendar = 583
    Camera = 40
    Can = 329
    CanOfBeer = 857
    CanOfFood = 280
    CanOpener = 279
    Candelabra = 605
    Candle = 137
    Candlestick = 148
    Cane = 555
    Canister = 794
    Cannister = 355
    CansOfCatFood = 593
    CapStand = 441
    Car = 530
    CardboardSheet = 452
    CardboardTube = 413
    Cart = 305
    Carton = 397
    Case = 851
    CasseroleDish = 365
    Cat = 594
    CatBed = 608
    CatCage = 580
    CatHouse = 856
    Cd = 207
    CdDisc = 585
    Ceiling = 4
    Celery = 720
    CellPhone = 290
    CellPhoneCharger = 602
    Centerpiece = 878
    CeramicFrog = 643
    Certificate = 790
    Chair = 5
    Chalkboard = 428
    Chandelier = 342
    Chapstick = 726
    Charger = 743
    ChargerAndWire = 574
    Chart = 411
    ChartRoll = 495
    ChartStand = 393
    Chessboard = 198
    Chest = 344
    ChildCarrier = 491
    Chimney = 702
    CircuitBreakerBox = 112
    ClassroomBoard = 392
    Cleaner = 548
    CleaningWipes = 381
    Clipboard = 536
    Clock = 56
    ClothBag = 492
    ClothDryingStand = 549
    Clothes = 141
    ClothingDetergent = 501
    ClothingDryer = 498
    ClothingDryingRack = 556
    ClothingHamper = 770
    ClothingHanger = 214
    ClothingIron = 572
    ClothingWasher = 499
    Coaster = 387
    CoatHanger = 400
    CoffeeBag = 226
    CoffeeGrinder = 237
    CoffeeMachine = 234
    CoffeePacket = 227
    CoffeePot = 893
    CoffeeTable = 356
    Coins = 308
    CokeBottle = 297
    Collander = 694
    Cologne = 176
    Column = 94
    Comb = 809
    Comforter = 484
    Computer = 46
    ComputerDisk = 616
    ConchShell = 673
    Cone = 6
    ConsoleController = 613
    ConsoleSystem = 518
    ContactLensCase = 634
    ContactLensSolutionBottle = 173
    Container = 140
    ContainerOfSkinCream = 637
    CookingPan = 252
    CookingPotCover = 761
    CopperVessel = 528
    CordlessPhone = 474
    CordlessTelephone = 545
    CorkBoard = 34
    Corkscrew = 713
    Corn = 716
    Counter = 7
    Cradle = 493
    Crate = 183
    Crayon = 511
    Cream = 635
    CreamTube = 653
    Crib = 485
    CrockPot = 330
    Cup = 35
    Curtain = 89
    CurtainRod = 582
    CuttingBoard = 247
    Decanter = 345
    DecorationItem = 842
    DecorativeBottle = 767
    DecorativeBowl = 826
    DecorativeCandle = 865
    DecorativeDish = 757
    DecorativeEgg = 862
    DecorativeItem = 853
    DecorativePlate = 383
    DecorativePlatter = 370
    Deoderant = 159
    Desk = 36
    DeskDrawer = 475
    DeskMat = 473
    Desser = 829
    DishBrush = 248
    DishCover = 368
    DishRack = 581
    DishScrubber = 261
    Dishes = 733
    Dishwasher = 8
    DisplayBoard = 444
    DisplayCase = 540
    DisplayPlatter = 877
    Dog = 701
    DogBed = 858
    DogBowl = 697
    DogCage = 703
    DogToy = 736
    Doily = 892
    Doll = 99
    DollHouse = 486
    DollarBill = 810
    Dolly = 219
    Door = 28
    DoorWindowReflection = 642
    DoorCurtain = 663
    DoorFacingTrimreflection = 657
    DoorFrame = 615
    DoorKnob = 27
    DoorLock = 646
    DoorWay = 609
    DoorWayArch = 686
    Doorreflection = 658
    Drain = 567
    Drawer = 174
    DrawerHandle = 371
    DrawerKnob = 833
    Dresser = 169
    Drum = 145
    DryingRack = 262
    DryingStand = 554
    Duck = 887
    Duster = 115
    Dvd = 197
    DvdPlayer = 170
    Dvds = 325
    Earplugs = 152
    EducationalDisplay = 419
    Eggplant = 888
    Eggs = 699
    ElectricBox = 550
    ElectricMixer = 369
    ElectricToothbrush = 142
    ElectricToothbrushBase = 629
    ElectricalKettle = 738
    ElectricalOutlet = 98
    ElectronicDrumset = 816
    Envelope = 476
    Envelopes = 843
    Eraser = 100
    EthernetJack = 118
    ExcerciseBall = 155
    ExcerciseEquipment = 457
    ExcerciseMachine = 558
    ExitSign = 86
    EyeGlasses = 335
    EyeballPlasticBall = 787
    FaceWashCream = 665
    Fan = 74
    Faucet = 9
    FaucetHandle = 568
    FaxMachine = 68
    FiberglassCase = 543
    Figurine = 836
    File = 75
    FileBox = 734
    FileContainer = 469
    FileHolder = 410
    FilePad = 619
    FileStand = 479
    FilingShelves = 401
    FireAlarm = 338
    FireExtinguisher = 10
    Fireplace = 372
    FishTank = 782
    Flag = 405
    Flashcard = 201
    Flashlight = 666
    Flask = 693
    FlaskSet = 760
    FlatbedScanner = 537
    Flipboard = 106
    Floor = 11
    FloorMat = 143
    FloorTrim = 868
    Flower = 81
    FlowerBasket = 595
    FlowerBox = 471
    FlowerPot = 146
    Folder = 69
    Folders = 213
    FoodProcessor = 715
    FoodWrappedOnATray = 752
    FoosballTable = 510
    FootRest = 163
    Football = 166
    Fork = 349
    FramedCertificate = 544
    Fruit = 286
    FruitBasket = 728
    FruitPlatter = 596
    FruitStand = 681
    Fruitplate = 682
    FryingPan = 318
    Furnace = 551
    Furniture = 524
    GameSystem = 516
    GameTable = 429
    GarageDoor = 850
    GarbageBag = 269
    GarbageBin = 12
    Garlic = 763
    Gate = 223
    GiftWrapping = 351
    GiftWrappingRoll = 185
    Glass = 612
    GlassBakingDish = 316
    GlassBox = 622
    GlassContainer = 636
    GlassDish = 721
    GlassPane = 412
    GlassPot = 304
    GlassRack = 216
    GlassSet = 705
    GlassWare = 889
    Globe = 347
    GlobeStand = 466
    Glove = 729
    GoldPiece = 880
    GrandfatherClock = 462
    Grapefruit = 597
    GreenScreen = 57
    Grill = 700
    Guitar = 300
    GuitarCase = 771
    HairBrush = 120
    HairDryer = 577
    HamburgerBun = 748
    Hammer = 883
    HandBlender = 599
    HandFan = 845
    HandSanitizer = 76
    HandSanitizerDispenser = 505
    HandSculpture = 309
    HandWeight = 838
    Handle = 758
    Hanger = 211
    Hangers = 209
    HangingHooks = 96
    Hat = 193
    HeadPhone = 586
    HeadPhones = 584
    Headband = 802
    Headboard = 161
    Headphones = 160
    Heater = 111
    HeatingTray = 714
    HockeyGlove = 194
    HockeyStick = 195
    HolePuncher = 61
    Hookah = 187
    Hooks = 95
    HoolaHoop = 512
    HorseToy = 513
    HotDogs = 722
    HotWaterHeater = 228
    Humidifier = 340
    IdCard = 478
    IncenseCandle = 644
    IncenseHolder = 672
    IndoorFountain = 863
    Inkwell = 824
    Ipad = 386
    Iphone = 296
    Ipod = 310
    IpodDock = 817
    IronBox = 557
    IronGrill = 463
    IroningBoard = 313
    Jacket = 324
    Jar = 70
    Jeans = 849
    Jersey = 311
    Jug = 687
    Juicer = 746
    KarateBelts = 775
    Key = 378
    Keyboard = 47
    KichenTowel = 264
    Kinect = 823
    KitchenContainerPlastic = 739
    KitchenIsland = 456
    KitchenItems = 253
    KitchenUtensil = 266
    KitchenUtensils = 753
    Kiwi = 598
    Knife = 259
    KnifeRack = 258
    Knob = 652
    Knobs = 600
    Label = 759
    Ladder = 48
    Ladel = 254
    Lamp = 144
    LampShade = 859
    Laptop = 37
    LaundryBasket = 164
    LaundryDetergentJug = 500
    LazySusan = 679
    Lectern = 882
    LegOfAGirl = 409
    Lego = 805
    Lemon = 765
    LetterStand = 620
    Lid = 533
    LidOfJar = 445
    LifeJacket = 784
    Light = 62
    LightBulb = 566
    LightSwitch = 301
    LightSwitchreflection = 659
    LightingTrack = 354
    LintComb = 798
    LintRoller = 178
    LitterBox = 606
    Lock = 180
    Luggage = 783
    LuggageRack = 803
    LunchBag = 407
    Machine = 220
    Magazine = 71
    MagazineHolder = 468
    Magic8Ball = 839
    Magnet = 23
    MailShelf = 65
    MailTray = 618
    Mailshelf = 153
    MakeupBrush = 121
    ManillaEnvelope = 63
    Mantel = 58
    Mantle = 874
    Map = 107
    Mask = 191
    Matchbox = 884
    Mattress = 576
    MeasuringCup = 730
    Medal = 776
    MedicineTube = 660
    Mellon = 707
    Menorah = 336
    MensSuit = 167
    MensTie = 315
    Mezuza = 531
    Microphone = 818
    MicrophoneStand = 821
    Microwave = 13
    MiniDisplayPlatform = 869
    Mirror = 122
    ModelBoat = 789
    Modem = 91
    Money = 482
    Monitor = 49
    MortarAndPestle = 357
    MotionCamera = 52
    Mouse = 103
    MousePad = 539
    Muffins = 229
    MugHanger = 749
    MugHolder = 744
    MusicKeyboard = 819
    MusicStand = 820
    MusicStereo = 442
    Nailclipper = 569
    Napkin = 244
    NapkinDispenser = 230
    NapkinHolder = 235
    NapkinRing = 350
    Necklace = 341
    NecklaceHolder = 779
    Newspapers = 873
    NightStand = 158
    Notebook = 210
    Notecards = 438
    OilContainer = 683
    Onion = 322
    Orange = 709
    OrangeJuicer = 745
    OrangePlasticCap = 704
    OrnamentalItem = 527
    OrnamentalPlant = 459
    OrnamentalPot = 735
    Ottoman = 359
    Oven = 238
    OvenHandle = 366
    OvenMitt = 754
    PackageOfBedroomSheets = 807
    PackageOfBottledWater = 875
    PackageOfWater = 684
    Pan = 589
    Paper = 15
    PaperBundle = 534
    PaperCutter = 108
    PaperHolder = 470
    PaperRack = 77
    PaperTowel = 113
    PaperTowelDispenser = 14
    PaperTowelHolder = 281
    PaperTray = 538
    PaperWeight = 480
    Papers = 483
    Peach = 710
    Pen = 97
    PenBox = 190
    PenCup = 786
    PenHolder = 464
    PenStand = 314
    Pencil = 396
    PencilHolder = 101
    Pepper = 885
    PepperGrinder = 579
    PepperShaker = 455
    Perfume = 655
    PerfumeBox = 654
    Person = 331
    PersonalCareLiquid = 649
    PhoneJack = 363
    Photo = 508
    PhotoAlbum = 864
    Piano = 298
    PianoBench = 460
    Picture = 64
    PictureOfFish = 394
    PieceOfWood = 552
    Pig = 811
    Pillow = 119
    Pineapple = 740
    PingPongBall = 623
    PingPongRacket = 624
    PingPongRacquet = 627
    PingPongTable = 625
    Pipe = 41
    Pitcher = 273
    PizzaBox = 274
    Placard = 420
    Placemat = 154
    Plant = 82
    PlantPot = 239
    Plaque = 231
    PlasticBowl = 320
    PlasticBox = 395
    PlasticChair = 489
    PlasticCrate = 402
    PlasticCupOfCoffee = 621
    PlasticDish = 723
    PlasticRack = 403
    PlasticToyContainer = 514
    PlasticTray = 404
    PlasticTub = 232
    Plate = 233
    Platter = 129
    Playpen = 815
    PoolBall = 520
    PoolSticks = 517
    PoolTable = 515
    PosterBoard = 406
    PosterCase = 116
    Pot = 16
    Potato = 323
    PowerSurge = 451
    Printer = 66
    Projector = 90
    ProjectorScreen = 53
    PuppyToy = 791
    Purse = 181
    Pyramid = 472
    Quill = 793
    Quilt = 575
    Radiator = 236
    Radio = 188
    Rags = 852
    Railing = 497
    RangeHood = 380
    Razor = 632
    ReflectionOfWindowShutters = 861
    Refridgerator = 17
    RemoteControl = 175
    RollOfPaperTowels = 449
    RollOfToiletPaper = 674
    RolledCarpet = 571
    RolledUpRug = 891
    RoomDivider = 87
    Rope = 560
    Router = 303
    Rug = 130
    Ruler = 72
    SaltAndPepper = 737
    SaltContainer = 361
    SaltShaker = 332
    Saucer = 217
    Scale = 639
    Scarf = 240
    Scenary = 832
    Scissor = 29
    Sculpture = 294
    SculptureOfTheChryslerBuilding = 846
    SculptureOfTheEiffelTower = 847
    SculptureOfTheEmpireStateBuilding = 848
    SecurityCamera = 212
    Server = 360
    ServingDish = 867
    ServingPlatter = 876
    ServingSpoon = 249
    SewingMachine = 890
    Shaver = 171
    ShavingCream = 570
    Sheet = 559
    SheetMusic = 461
    SheetOfMetal = 287
    Sheets = 348
    ShelfFrame = 855
    Shelves = 42
    ShirtsInHanger = 302
    Shoe = 149
    ShoeHanger = 834
    ShoeRack = 614
    Shoelace = 785
    Shofar = 546
    ShoppingBaskets = 222
    ShoppingCart = 319
    Shorts = 192
    Shovel = 607
    ShowPiece = 454
    ShowerBase = 667
    ShowerCap = 132
    ShowerCurtain = 123
    ShowerHead = 650
    ShowerHose = 669
    ShowerKnob = 651
    ShowerPipe = 664
    ShowerTube = 675
    Sifter = 727
    Sign = 208
    Sink = 24
    SinkProtector = 270
    SixPackOfBeer = 382
    SleepingBag = 841
    Slide = 814
    Soap = 133
    SoapBox = 671
    SoapDish = 638
    SoapHolder = 506
    SoapStand = 640
    SoapTray = 662
    SoccerBall = 837
    Sock = 165
    Sofa = 83
    SoftToy = 422
    SoftToyGroup = 421
    Spatula = 255
    Speaker = 54
    SpiceBottle = 272
    SpiceRack = 241
    SpiceStand = 256
    Sponge = 250
    Spoon = 283
    SpoonSets = 592
    SpoonStand = 282
    SpotLight = 353
    Squash = 717
    SqueezeTube = 131
    StackOfPlates = 358
    StackedBinsBoxes = 446
    StackedChairs = 43
    StackedPlasticRacks = 447
    Stairs = 215
    Stamp = 114
    Stand = 50
    StapleRemover = 202
    Stapler = 67
    Steamer = 742
    StepStool = 276
    Stereo = 84
    Stick = 529
    Sticker = 725
    Sticks = 561
    Stones = 578
    Stool = 150
    StorageBin = 812
    StorageChest = 813
    StorageRack = 448
    StorageShelvesbooks = 430
    StorageSpace = 645
    Stove = 242
    StoveBurner = 18
    Stroller = 373
    StuffedAnimal = 177
    StyrofoamObject = 44
    SugerJar = 741
    SuitJacket = 379
    Suitcase = 199
    SurgeProtect = 611
    SurgeProtector = 326
    Switchbox = 364
    Table = 19
    TableRunner = 375
    Tablecloth = 292
    Tag = 218
    Tape = 109
    TapeDispenser = 30
    TeaBox = 879
    TeaCannister = 769
    TeaCoaster = 711
    TeaKettle = 243
    TeaPot = 678
    Telephone = 32
    TelephoneCord = 31
    Telescope = 467
    Television = 172
    TennisRacket = 626
    Tent = 835
    Thermostat = 110
    Throw = 872
    TinFoil = 265
    Tissue = 648
    TissueBox = 138
    TissueRoll = 764
    Toaster = 251
    ToasterOven = 275
    Toilet = 124
    ToiletBowlBrush = 565
    ToiletBrush = 630
    ToiletPaper = 139
    ToiletPaperHolder = 647
    ToiletPlunger = 563
    Toiletries = 631
    ToiletriesBag = 125
    Toothbrush = 127
    ToothbrushHolder = 126
    Toothpaste = 128
    ToothpasteHolder = 670
    Torah = 894
    Torch = 696
    Towel = 135
    TowelRod = 134
    Toy = 389
    ToyApple = 830
    ToyBin = 417
    ToyBoat = 795
    ToyBottle = 182
    ToyBox = 434
    ToyCar = 415
    ToyCashRegister = 532
    ToyChair = 487
    ToyChest = 801
    ToyCube = 423
    ToyCuboid = 431
    ToyCylinder = 424
    ToyDog = 831
    ToyDoll = 465
    ToyHorse = 828
    ToyHouse = 490
    ToyKitchen = 751
    ToyPhone = 435
    ToyPlane = 481
    ToyPyramid = 788
    ToyRectangle = 425
    ToyShelf = 416
    ToySink = 436
    ToySofa = 488
    ToyStroller = 854
    ToyTable = 526
    ToyTree = 432
    ToyTriangle = 426
    ToyTruck = 391
    ToyTrucks = 439
    Toyhouse = 437
    ToysBasket = 390
    ToysBox = 496
    ToysRack = 443
    ToysShelf = 418
    TrackLight = 33
    Trampoline = 521
    TravelBag = 799
    Tray = 179
    Treadmill = 458
    TreeSculpture = 541
    Tricycle = 522
    Trinket = 844
    Trivet = 257
    Trolley = 504
    Trolly = 221
    Trophy = 547
    TubOfTupperware = 604
    Tumbler = 327
    TunaCans = 590
    Tupperware = 762
    TvStand = 291
    Typewriter = 376
    Umbrella = 203
    Unknown = 20
    Urn = 151
    UsbDrive = 587
    Utensil = 267
    UtensilContainer = 362
    Utensils = 317
    VacuumCleaner = 306
    Vase = 78
    Vasoline = 184
    Vegetable = 724
    VegetablePeeler = 277
    Vegetables = 719
    Vessel = 263
    VesselSet = 706
    Vessels = 601
    VhsTapes = 871
    VideoGame = 519
    Vuvuzela = 196
    WaffleMaker = 288
    WalkieTalkie = 398
    Walkietalkie = 866
    Wall = 21
    WallDecoration = 186
    WallDivider = 800
    WallHandSanitizerDispenser = 440
    WallStand = 295
    Wallet = 661
    Wardrobe = 772
    WashingMachine = 278
    Watch = 384
    WaterCarboy = 102
    WaterCooler = 509
    WaterDispenser = 507
    WaterFilter = 731
    WaterFountain = 339
    WaterHeater = 588
    WaterPurifier = 93
    Watermellon = 718
    Webcam = 781
    Whisk = 367
    Whiteboard = 45
    WhiteboardEraser = 388
    WhiteboardMarker = 117
    Wii = 523
    Window = 59
    WindowBox = 778
    WindowCover = 573
    WindowFrame = 477
    WindowSeat = 777
    WindowShelf = 668
    Wine = 766
    WineAccessory = 732
    WineBottle = 333
    WineGlass = 293
    WineRack = 299
    Wire = 92
    WireBasket = 603
    WireBoard = 792
    WireRack = 105
    WireTray = 768
    WoodenContainer = 321
    WoodenKitchenUtensils = 284
    WoodenPillar = 553
    WoodenPlank = 698
    WoodenPlanks = 562
    WoodenToy = 433
    WoodenUtensil = 756
    WoodenUtensils = 346
    Wreathe = 881
    Xbox = 628
    Yarmulka = 806
    YellowPepper = 886
    YogaMat = 205
    No_Label = 0

    @classmethod
    def label_frequencies(cls):
        return {
            cls.Wall: 0.21395508935760293,
            cls.No_Label: 0.17381115702999195,
            cls.Floor: 0.09063516325914424,
            cls.Cabinet: 0.062008917338825624,
            cls.Bed: 0.037681449221445826,
            cls.Chair: 0.03329489536605705,
            cls.Sofa: 0.026781670548654244,
            cls.Door: 0.021622758421756382,
            cls.Table: 0.021377611355532553,
            cls.Window: 0.021201466169197148,
            cls.Picture: 0.020844251423395445,
            cls.Bookshelf: 0.019332181317575338,
            cls.Blinds: 0.016721297554347828,
            cls.Counter: 0.014384997016620658,
            cls.Ceiling: 0.013741377853979756,
            cls.Desk: 0.011107662702007132,
            cls.Curtain: 0.011033868993702553,
            cls.Shelves: 0.010194517231711525,
            cls.Mirror: 0.009960661177967563,
            cls.Dresser: 0.009162389291465378,
            cls.Pillow: 0.00827010276483782,
            cls.FloorMat: 0.007351590356855302,
            cls.Clothes: 0.00700659892957787,
            cls.Books: 0.005907594317920405,
            cls.Refridgerator: 0.005825946773636991,
            cls.Television: 0.004940761498590982,
            cls.Paper: 0.0039016041954221302,
            cls.ShowerCurtain: 0.0037681878306878307,
            cls.Towel: 0.0036624800508971707,
            cls.Box: 0.0033747299682252126,
            cls.Whiteboard: 0.003179493849925236,
            cls.Person: 0.0029459680347078444,
            cls.NightStand: 0.002789529326978376,
            cls.Lamp: 0.0027702047568725556,
            cls.Sink: 0.002669466280624569,
            cls.Toilet: 0.002653884413819876,
            cls.Bathtub: 0.0025435892389866575,
            cls.Bag: 0.002302879050925926,
            cls.Bottle: 0.002202899898636991,
            cls.Oven: 0.0021844290063549572,
            cls.Dishwasher: 0.002150369507706464,
            cls.Book: 0.002076427529043018,
            cls.Blanket: 0.001960183567690361,
            cls.Column: 0.0018353893489763055,
            cls.Monitor: 0.0017648710676903612,
            cls.Stove: 0.001747741348199908,
            cls.GarbageBin: 0.0017374567769438693,
            cls.RoomDivider: 0.0016603472042500576,
            cls.Headboard: 0.0016456324943926845,
            cls.Piano: 0.0015661640750805153,
            cls.Microwave: 0.0015564995435070163,
            cls.Plant: 0.0014988920160455487,
            cls.Unknown: 0.0014145059631642512,
            cls.CoffeeTable: 0.0013824458678398895,
            cls.Printer: 0.0013419074038129744,
            cls.CorkBoard: 0.0011959599472337243,
            cls.KitchenIsland: 0.0011915949577294687,
            cls.Cup: 0.001009276332815735,
            cls.Ottoman: 0.001005320210058661,
            cls.Drawer: 0.0009977876265240397,
            cls.Light: 0.0009944425573671497,
            cls.MailShelf: 0.000771356323326432,
            cls.Fireplace: 0.0007686874568668047,
            cls.Bowl: 0.0007655378350011502,
            cls.Rug: 0.0007450900046008742,
            cls.BunkBed: 0.0007387548165401427,
            cls.Computer: 0.0006883698527720267,
            cls.WallDecoration: 0.0006727520416379112,
            cls.Mattress: 0.0006649768338796872,
            cls.Basket: 0.000652333416005291,
            cls.Blackboard: 0.0006505474321371061,
            cls.Stool: 0.0006272869579595123,
            cls.Chandelier: 0.000600751056763285,
            cls.Bench: 0.0005890354518921095,
            cls.StuffedAnimal: 0.0005884086726478031,
            cls.Backpack: 0.0005807705026455027,
            cls.TvStand: 0.0005785846381124914,
            cls.CoffeeMachine: 0.0005666398881412468,
            cls.Toy: 0.0005523475241545893,
            cls.ClassroomBoard: 0.0005405398119392685,
            cls.RangeHood: 0.0005327443854957442,
            cls.ProjectorScreen: 0.0005226193172015183,
            cls.Banister: 0.0005222486413043478,
            cls.AirConditioner: 0.0005070396969174143,
            cls.AirDuct: 0.0005033554031515988,
            cls.Faucet: 0.0005005652245801702,
            cls.DryingRack: 0.0004884115481941569,
            cls.GlassRack: 0.0004875106934380032,
            cls.Placemat: 0.0004784864202323441,
            cls.TissueBox: 0.0004750200389636531,
            cls.Flower: 0.0004623159650333563,
            cls.Vase: 0.0004620755873303428,
            cls.LaundryBasket: 0.0004615701201978376,
            cls.Jar: 0.000458499126552795,
            cls.Fan: 0.00044912664265585463,
            cls.Container: 0.000400570346647113,
            cls.PaperTowel: 0.0004002064103117092,
            cls.Crib: 0.00039524609285139175,
            cls.Bicycle: 0.0003915865108120543,
            cls.Wardrobe: 0.0003520252832413158,
            cls.Speaker: 0.0003474199160340465,
            cls.Sculpture: 0.00033902691511387165,
            cls.Purse: 0.00032005055569933285,
            cls.TableRunner: 0.0003124213717793881,
            cls.SpiceRack: 0.00030840908600759145,
            cls.File: 0.00030146284434667586,
            cls.Laptop: 0.0002938246743443754,
            cls.Bin: 0.00028369061996779386,
            cls.Tray: 0.0002809947952610996,
            cls.Pipe: 0.000278002429836669,
            cls.PianoBench: 0.00027769465651598805,
            cls.ToasterOven: 0.0002637505032206119,
            cls.GreenScreen: 0.00025833638860133423,
            cls.PingPongTable: 0.00025805332700713134,
            cls.Map: 0.00025627183618012425,
            cls.Telephone: 0.00025451505707959514,
            cls.Magazine: 0.00025446338710605014,
            cls.Heater: 0.0002539466873706004,
            cls.Doll: 0.00024964235392224525,
            cls.Tent: 0.00024669716543018175,
            cls.BabyChair: 0.00024629953128594434,
            cls.Railing: 0.00024139762336093857,
            cls.Furniture: 0.00023837605316885208,
            cls.CuttingBoard: 0.00023435927435587763,
            cls.ExcerciseEquipment: 0.0002320116603404647,
            cls.Treadmill: 0.00022766464300092018,
            cls.PoolTable: 0.00022624933502990568,
            cls.WashingMachine: 0.0002231334109730849,
            cls.GarageDoor: 0.0002219090572521279,
            cls.Stairs: 0.0002200399521221532,
            cls.Jersey: 0.00021725875963308027,
            cls.Stand: 0.00021623209972394754,
            cls.Plate: 0.00021600070810328962,
            cls.Keyboard: 0.00021567945565907523,
            cls.Stereo: 0.0002141226168909593,
            cls.Shoe: 0.0002132936507936508,
            cls.Playpen: 0.0002132554599436393,
            cls.PlantPot: 0.00020251709152864045,
            cls.Hat: 0.00020096699232229125,
            cls.Ladder: 0.00019571687370600413,
            cls.Machine: 0.00018545476765585462,
            cls.AirVent: 0.00018302852541982978,
            cls.FoosballTable: 0.00017858266117437314,
            cls.ElectricalOutlet: 0.0001784411303772717,
            cls.Wire: 0.00017001667816885207,
            cls.IroningBoard: 0.0001674331794916034,
            cls.GlassPane: 0.0001641667385553255,
            cls.Luggage: 0.0001566656063089487,
            cls.WaffleMaker: 0.00015353171008741662,
            cls.DecorativePlate: 0.0001513907759661836,
            cls.Gate: 0.00014974182985392223,
            cls.Clock: 0.00014948123346560847,
            cls.Candlestick: 0.00014757169096503336,
            cls.Quilt: 0.00014576099537037037,
            cls.Urn: 0.0001434088883137796,
            cls.Desser: 0.00014279334167241776,
            cls.ToiletPaper: 0.00014005483307453416,
            cls.Board: 0.0001390416522889349,
            cls.KnifeRack: 0.00013802847150333564,
            cls.Mantel: 0.0001347732631700023,
            cls.DisplayBoard: 0.00013106650419829766,
            cls.CrockPot: 0.0001299769417126754,
            cls.Globe: 0.00012922885035656774,
            cls.ToyKitchen: 0.00012683630592937658,
            cls.Mask: 0.00012671724033816426,
            cls.Napkin: 0.00012594668377616746,
            cls.Guitar: 0.00012573326432022085,
            cls.Folder: 0.00012424157464918333,
            cls.Throw: 0.00012322390082240625,
            cls.ToyHouse: 0.00012256342376926616,
            cls.WaterPurifier: 0.00012251624683689901,
            cls.Chessboard: 0.0001220939009661836,
            cls.Notebook: 0.00012197932841614907,
            cls.TowelRod: 0.0001219680958132045,
            cls.TeaKettle: 0.00012122449749827468,
            cls.Stroller: 0.00011936662497124454,
            cls.Blender: 0.00011853990539452496,
            cls.FlowerPot: 0.00011565312643777317,
            cls.Papers: 0.000114637699131585,
            cls.ShirtsInHanger: 0.00011421984630204739,
            cls.Jacket: 0.00011415694372555786,
            cls.PosterBoard: 0.00011229907119852772,
            cls.GarbageBag: 0.00011185426012192316,
            cls.Suitcase: 0.00011033336568322982,
            cls.Pitcher: 0.00011020531400966184,
            cls.DecorativePlatter: 0.00010954932999769957,
            cls.Sheets: 0.00010728933028525419,
            cls.Pot: 0.00010683104008511617,
            cls.GuitarCase: 0.00010291535469864274,
            cls.ToyShelf: 0.00010180782004830917,
            cls.Flipboard: 0.00010065760150678628,
            cls.ExcerciseMachine: 9.912547446514837e-05,
            cls.SpiceBottle: 9.906481840924776e-05,
            cls.CatHouse: 9.749225399700943e-05,
            cls.DvdPlayer: 9.63914589084426e-05,
            cls.ShowPiece: 9.367316899585921e-05,
            cls.DogCage: 9.298124065447434e-05,
            cls.Hanger: 9.213879543363239e-05,
            cls.Dog: 8.951261286519439e-05,
            cls.Comforter: 8.949464070048309e-05,
            cls.SuitJacket: 8.622370672302738e-05,
            cls.ToysRack: 8.26966693984357e-05,
            cls.DoorFrame: 8.03153575741891e-05,
            cls.AlarmClock: 7.940551673567977e-05,
            cls.Flag: 7.794977139406487e-05,
            cls.CableBox: 7.748474163216011e-05,
            cls.Ball: 7.64243839141937e-05,
            cls.Can: 7.63480022141707e-05,
            cls.ClothingDryer: 7.430816151943869e-05,
            cls.DoorKnob: 7.396669038992408e-05,
            cls.Scale: 7.349492106625259e-05,
            cls.Toyhouse: 7.300967261904762e-05,
            cls.FaucetHandle: 7.248398680124223e-05,
            cls.GrandfatherClock: 7.233571644237405e-05,
            cls.Chest: 7.175611413043478e-05,
            cls.MensSuit: 7.132028913618587e-05,
            cls.PictureOfFish: 7.051603476535542e-05,
            cls.RemoteControl: 6.994541853577179e-05,
            cls.Server: 6.978591557395905e-05,
            cls.Sheet: 6.873454393834828e-05,
            cls.WindowShelf: 6.859301314124684e-05,
            cls.Binder: 6.457398780768346e-05,
            cls.DryingStand: 6.415388845755694e-05,
            cls.DoorCurtain: 6.353384877501725e-05,
            cls.Toaster: 6.267792443064183e-05,
            cls.BabyGate: 6.224209943639291e-05,
            cls.WindowSeat: 6.219941554520359e-05,
            cls.WindowFrame: 6.134798424200598e-05,
            cls.Calendar: 6.115702999194847e-05,
            cls.Radiator: 6.074367020358868e-05,
            cls.FootRest: 6.0451622527030137e-05,
            cls.WindowBox: 5.9007109788359786e-05,
            cls.Cd: 5.8863332470669426e-05,
            cls.StepStool: 5.866339213825627e-05,
            cls.BakingDish: 5.743679189671037e-05,
            cls.Sponge: 5.726605633195307e-05,
            cls.WineRack: 5.6994227340694735e-05,
            cls.PosterCase: 5.652919757878997e-05,
            cls.FryingPan: 5.595184178743961e-05,
            cls.PlasticBox: 5.594510222567288e-05,
            cls.Chart: 5.52127365136876e-05,
            cls.MailTray: 5.518128522544283e-05,
            cls.ClothingWasher: 5.347842261904762e-05,
            cls.WoodenPlanks: 5.269663345410628e-05,
            cls.StorageRack: 5.267416824821716e-05,
            cls.Tablecloth: 5.243154402461468e-05,
            cls.Drum: 5.192158385093168e-05,
            cls.SheetMusic: 5.096456608005521e-05,
            cls.PackageOfBedroomSheets: 5.0692737088796874e-05,
            cls.RolledCarpet: 4.983681274442144e-05,
            cls.Stapler: 4.963911893259719e-05,
            cls.SheetOfMetal: 4.936728994133885e-05,
            cls.Shorts: 4.9279675638371294e-05,
            cls.HotWaterHeater: 4.919206133540373e-05,
            cls.Sign: 4.897190231769036e-05,
            cls.ExcerciseBall: 4.8931464947089946e-05,
            cls.BeddingPackage: 4.892023234414539e-05,
            cls.DeskDrawer: 4.792951676443524e-05,
            cls.WoodenPillar: 4.7590292155509546e-05,
            cls.ElectricToothbrush: 4.7271286231884056e-05,
            cls.Trophy: 4.678603778467909e-05,
            cls.Scarf: 4.6237886760984586e-05,
            cls.WindowCover: 4.617947722567288e-05,
            cls.SinkProtector: 4.5900908672647805e-05,
            cls.Dvds: 4.58245269726248e-05,
            cls.StoveBurner: 4.485852311939268e-05,
            cls.DishRack: 4.399585921325052e-05,
            cls.FishTank: 4.391498447204969e-05,
            cls.Candle: 4.3375819530710835e-05,
            cls.Sticks: 4.3351107804232804e-05,
            cls.Bucket: 4.3103990539452494e-05,
            cls.FruitStand: 4.21604518921095e-05,
            cls.DoorWay: 4.1457290947780076e-05,
            cls.Folders: 4.0657529618127444e-05,
            cls.SpiceStand: 4.0585640959282264e-05,
            cls.ToysShelf: 4.014307640326662e-05,
            cls.Banana: 4.012735075914424e-05,
            cls.Barrel: 3.936802680009202e-05,
            cls.ElectricMixer: 3.890973659995399e-05,
            cls.PaperCutter: 3.8723275391074306e-05,
            cls.VacuumCleaner: 3.6696913819875777e-05,
            cls.Chimney: 3.645878263745112e-05,
            cls.Cart: 3.6112818466758685e-05,
            cls.Collander: 3.5198484587071544e-05,
            cls.FaxMachine: 3.491317647227973e-05,
            cls.Glass: 3.4531267972164714e-05,
            cls.WaterFountain: 3.449757016333103e-05,
            cls.Umbrella: 3.4466118875086266e-05,
            cls.Basketball: 3.422798769266161e-05,
            cls.Projector: 3.38303535484242e-05,
            cls.NapkinHolder: 3.366186450425581e-05,
            cls.Trivet: 3.363265973659995e-05,
            cls.HolePuncher: 3.195450885668277e-05,
            cls.Xbox: 3.1887113239015415e-05,
            cls.Thermostat: 3.140860435357718e-05,
            cls.DollHouse: 3.119293837704164e-05,
            cls.ToyTable: 3.11480079652634e-05,
            cls.ShoeHanger: 3.0869439412238324e-05,
            cls.WaterCooler: 3.068073168276973e-05,
            cls.CoffeePacket: 3.0593117379802164e-05,
            cls.Menorah: 3.039991660915574e-05,
            cls.ManillaEnvelope: 3.029208362088797e-05,
            cls.KitchenItems: 2.998431030020704e-05,
            cls.ToothbrushHolder: 2.9892202956061653e-05,
            cls.ShoppingCart: 2.9548485305958132e-05,
            cls.LightSwitch: 2.9231725902921555e-05,
            cls.MagazineHolder: 2.918904201173223e-05,
            cls.Mouse: 2.9186795491143316e-05,
            cls.Vessel: 2.8957650391074304e-05,
            cls.SleepingBag: 2.889474781458477e-05,
            cls.ReflectionOfWindowShutters: 2.8281447693811824e-05,
            cls.ClothingHamper: 2.7865841384863123e-05,
            cls.TubOfTupperware: 2.764792888773867e-05,
            cls.Knife: 2.731544384057971e-05,
            cls.ToyChest: 2.7283992552334945e-05,
            cls.IndoorFountain: 2.701890312284334e-05,
            cls.TapeDispenser: 2.6942521422820337e-05,
            cls.Chalkboard: 2.658981769036117e-05,
            cls.PlasticCrate: 2.650894294916034e-05,
            cls.Trampoline: 2.6100076201978376e-05,
            cls.PaperHolder: 2.5412640901771337e-05,
            cls.Deoderant: 2.5374450051759834e-05,
            cls.SqueezeTube: 2.533176616057051e-05,
            cls.Saucer: 2.527560314584771e-05,
            cls.Apple: 2.521494708994709e-05,
            cls.BagOfChips: 2.5041965004600876e-05,
            cls.Celery: 2.4916159851621807e-05,
            cls.TravelBag: 2.4774629054520357e-05,
            cls.ClothingDryingRack: 2.459715392799632e-05,
            cls.CircuitBreakerBox: 2.442192532206119e-05,
            cls.Antenna: 2.440619967793881e-05,
            cls.HandFan: 2.392095123073384e-05,
            cls.Broom: 2.3786159995399127e-05,
            cls.Toothbrush: 2.3557014895330112e-05,
            cls.Cream: 2.335258152173913e-05,
            cls.Tricycle: 2.3309897630549805e-05,
            cls.Platter: 2.31391620657925e-05,
            cls.Crate: 2.306727340694732e-05,
            cls.GlobeStand: 2.3024589515757995e-05,
            cls.FireExtinguisher: 2.298190562456867e-05,
            cls.PaperTowelDispenser: 2.2680871865654474e-05,
            cls.Bread: 2.2620215809753853e-05,
            cls.HairBrush: 2.261572276857603e-05,
            cls.WaterDispenser: 2.242701503910743e-05,
            cls.ToyTruck: 2.2415782436162873e-05,
            cls.WineGlass: 2.228548424200598e-05,
            cls.DeskMat: 2.2170911691971476e-05,
            cls.CatCage: 2.1625007188865884e-05,
            cls.Briefcase: 2.1332959512307338e-05,
            cls.WaterCarboy: 2.1150991344605475e-05,
            cls.OrnamentalPlant: 2.09285858063032e-05,
            cls.DoorWindowReflection: 2.0793794570968485e-05,
            cls.ElectricalKettle: 2.0760096762134805e-05,
            cls.Brick: 2.055341686795491e-05,
            cls.GameTable: 2.0277094835518747e-05,
            cls.BasketballHoop: 2.0274848314929837e-05,
            cls.SewingMachine: 2.0061428858983207e-05,
            cls.ToyHorse: 1.998280063837129e-05,
            cls.Candelabra: 1.9787353347135956e-05,
            cls.ChartStand: 1.9722204250057512e-05,
            cls.YogaMat: 1.9713218167701863e-05,
            cls.Shaver: 1.9681766879457096e-05,
            cls.CurtainRod: 1.967502731769036e-05,
            cls.Dvd: 1.95874130147228e-05,
            cls.LetterStand: 1.923470928226363e-05,
            cls.FileBox: 1.9081945882217623e-05,
            cls.NapkinDispenser: 1.903476894985047e-05,
            cls.Steamer: 1.886627990568208e-05,
            cls.Furnace: 1.877417256153669e-05,
            cls.ShelfFrame: 1.8612423079135037e-05,
            cls.ToyBox: 1.8574232229123534e-05,
            cls.MusicStand: 1.84731388026225e-05,
            cls.FileHolder: 1.841922230848861e-05,
            cls.MousePad: 1.8293417155509547e-05,
            cls.Hangers: 1.7819401311249138e-05,
            cls.Doorreflection: 1.7803675667126753e-05,
            cls.DishCover: 1.7794689584771107e-05,
            cls.DogBowl: 1.7767731337704163e-05,
            cls.Humidifier: 1.7455464975845412e-05,
            cls.StyrofoamObject: 1.7365604152288934e-05,
            cls.Radio: 1.7322920261099608e-05,
            cls.PaperTray: 1.7280236369910282e-05,
            cls.Fruitplate: 1.7185882505175984e-05,
            cls.GameSystem: 1.7127472969864273e-05,
            cls.SaltShaker: 1.7044351708074534e-05,
            cls.WaterFilter: 1.7037612146307798e-05,
            cls.Rags: 1.6855643978605935e-05,
            cls.Medal: 1.6839918334483553e-05,
            cls.Vessels: 1.6428805066712675e-05,
            cls.HockeyStick: 1.6372642051989877e-05,
            cls.MusicStereo: 1.636590249022314e-05,
            cls.ToyTree: 1.6022184840119624e-05,
            cls.Pan: 1.599522659305268e-05,
            cls.ClothingIron: 1.5932324016563145e-05,
            cls.ExitSign: 1.562904373706004e-05,
            cls.TunaCans: 1.5611071572348746e-05,
            cls.ToySink: 1.56020854899931e-05,
            cls.ToyCar: 1.55077316252588e-05,
            cls.ToyTrucks: 1.5184232660455486e-05,
            cls.Lego: 1.5179739619277663e-05,
            cls.TeaPot: 1.5161767454566368e-05,
            cls.Cables: 1.504719490453186e-05,
            cls.OrnamentalItem: 1.503371578099839e-05,
            cls.ShowerHead: 1.5022483178053831e-05,
            cls.StorageChest: 1.4984292328042328e-05,
            cls.LazySusan: 1.4907910628019323e-05,
            cls.Switchbox: 1.4730435501495283e-05,
            cls.Toothpaste: 1.4613616430871866e-05,
            cls.Pen: 1.4539481251437773e-05,
            cls.CoffeeBag: 1.4382224810213941e-05,
            cls.Decanter: 1.4267652260179434e-05,
            cls.CokeBottle: 1.4159819271911664e-05,
            cls.WallDivider: 1.3804869018863584e-05,
            cls.IronGrill: 1.3715008195307109e-05,
            cls.Grill: 1.3699282551184725e-05,
            cls.ToiletriesBag: 1.3622900851161721e-05,
            cls.Fruit: 1.3586956521739131e-05,
            cls.BeanBag: 1.3557751754083275e-05,
            cls.MusicKeyboard: 1.3279183201058202e-05,
            cls.ElectronicDrumset: 1.3252224953991259e-05,
            cls.SoftToy: 1.3184829336323902e-05,
            cls.ToySofa: 1.2971409880377272e-05,
            cls.CoatHanger: 1.29579307568438e-05,
            cls.MensTie: 1.2784948671497585e-05,
            cls.ClothingDetergent: 1.2742264780308259e-05,
            cls.Lectern: 1.2690594806763285e-05,
            cls.DogBed: 1.2605227024384633e-05,
            cls.PaperRack: 1.2582761818495514e-05,
            cls.SpoonStand: 1.2528845324361629e-05,
            cls.FruitBasket: 1.2519859242005981e-05,
            cls.FilePad: 1.2468189268461008e-05,
            cls.Camera: 1.2248030250747642e-05,
            cls.ToiletBrush: 1.2113239015412928e-05,
            cls.HotDogs: 1.209077380952381e-05,
            cls.Mantle: 1.198518734184495e-05,
            cls.Pencil: 1.1809958735909823e-05,
            cls.GlassBox: 1.1798726132965264e-05,
            cls.CellPhone: 1.1780753968253969e-05,
            cls.PenStand: 1.1753795721187025e-05,
            cls.GlassBakingDish: 1.1688646624108581e-05,
            cls.Case: 1.1502185415228894e-05,
            cls.CeramicFrog: 1.1493199332873246e-05,
            cls.WaterHeater: 1.1457255003450656e-05,
            cls.Dolly: 1.1434789797561537e-05,
            cls.Plaque: 1.1315724206349206e-05,
            cls.ServingDish: 1.1248328588681849e-05,
            cls.FlatbedScanner: 1.1050634776857603e-05,
            cls.ShoppingBaskets: 1.095178787094548e-05,
            cls.DecorativeBowl: 1.0936062226823097e-05,
            cls.BedSheets: 1.092707614446745e-05,
            cls.WalkieTalkie: 1.0902364417989418e-05,
            cls.Watermellon: 1.0783298826777088e-05,
            cls.Slide: 1.0767573182654704e-05,
            cls.HairDryer: 1.0747354497354498e-05,
            cls.Utensils: 1.0659740194386933e-05,
            cls.Soap: 1.0648507591442374e-05,
            cls.KarateBelts: 1.0623795864964343e-05,
            cls.BagOfFlour: 1.0617056303197607e-05,
            cls.ToyStroller: 1.0603577179664136e-05,
            cls.PencilHolder: 1.0471032464918334e-05,
            cls.Muffins: 1.0453060300207039e-05,
            cls.Pineapple: 1.0430595094317921e-05,
            cls.StorageShelvesbooks: 1.0381171641361859e-05,
            cls.Belt: 1.034747383252818e-05,
            cls.PlasticTub: 1.0282324735449735e-05,
            cls.ToyChair: 1.0235147803082586e-05,
            cls.Tissue: 1.020594303542673e-05,
            cls.Trolly: 1.0152026541292846e-05,
            cls.PizzaBox: 1.0143040458937198e-05,
            cls.Glove: 1.0118328732459167e-05,
            cls.CatBed: 1.0113835691281344e-05,
            cls.LifeJacket: 1.0032960950080515e-05,
            cls.ShowerBase: 1.0032960950080515e-05,
            cls.ToyBoat: 9.976797935357719e-06,
            cls.Spoon: 9.927374482401657e-06,
            cls.TennisRacket: 9.889183632390154e-06,
            cls.CapStand: 9.8599788647343e-06,
            cls.HeadPhones: 9.855485823556475e-06,
            cls.Scissor: 9.830774097078444e-06,
            cls.Envelope: 9.680257217621348e-06,
            cls.SoapDish: 9.615108120542903e-06,
            cls.AmericanFlag: 9.583656832298137e-06,
            cls.Flask: 9.424153870485392e-06,
            cls.NapkinRing: 9.367990855762594e-06,
            cls.Tupperware: 9.320813923395445e-06,
            cls.SpoonSets: 9.311827841039798e-06,
            cls.Spatula: 9.260157867494824e-06,
            cls.Tag: 9.179283126293995e-06,
            cls.Necklace: 9.091668823326432e-06,
            cls.CoffeeGrinder: 9.069203617437312e-06,
            cls.ShoeRack: 9.010794082125605e-06,
            cls.CookingPan: 9.004054520358868e-06,
            cls.IpodDock: 8.981589314469749e-06,
            cls.Calculator: 8.860277202668507e-06,
            cls.LaundryDetergentJug: 8.651350787899702e-06,
            cls.ElectricBox: 8.633378623188405e-06,
            cls.Sticker: 8.597434293765815e-06,
            cls.Stones: 8.595187773176904e-06,
            cls.Bassinet: 8.568229526109961e-06,
            cls.BarOfSoap: 8.444670893719806e-06,
            cls.LuggageRack: 8.428945249597424e-06,
            cls.Shoelace: 8.422205687830688e-06,
            cls.ConsoleController: 8.350317028985507e-06,
            cls.Flashcard: 8.350317028985507e-06,
            cls.StackedChairs: 8.339084426040948e-06,
            cls.PlasticChair: 8.334591384863125e-06,
            cls.WireRack: 8.314372699562917e-06,
            cls.CableRack: 8.237990999539912e-06,
            cls.Canister: 8.18856754658385e-06,
            cls.ToyBin: 8.053776311249138e-06,
            cls.FileContainer: 7.862822061191626e-06,
            cls.PackageOfBottledWater: 7.84484989648033e-06,
            cls.Carton: 7.833617293535772e-06,
            cls.Typewriter: 7.734770387623648e-06,
            cls.FaceWashCream: 7.696579537612146e-06,
            cls.WireBasket: 7.678607372900851e-06,
            cls.PenCup: 7.656142167011732e-06,
            cls.PlasticDish: 7.65389564642282e-06,
            cls.Ladel: 7.613458275822406e-06,
            cls.Headphones: 7.561788302277432e-06,
            cls.Clipboard: 7.559541781688521e-06,
            cls.GiftWrapping: 7.47642051989878e-06,
            cls.WineBottle: 7.471927478720957e-06,
            cls.Jeans: 7.4337366287094544e-06,
            cls.StorageSpace: 7.303438434552565e-06,
            cls.BottleOfListerine: 7.263001063952151e-06,
            cls.Onion: 7.247275419829768e-06,
            cls.SoftToyGroup: 7.2337962962962966e-06,
            cls.NecklaceHolder: 7.132702869795261e-06,
            cls.Bookend: 7.128209828617437e-06,
            cls.Football: 7.121470266850701e-06,
            cls.SoapHolder: 7.1169772256728775e-06,
            cls.Potato: 7.114730705083966e-06,
            cls.DishScrubber: 7.103498102139406e-06,
            cls.Scenary: 7.092265499194847e-06,
            cls.CookingPotCover: 7.029362922705314e-06,
            cls.ToysBox: 7.027116402116402e-06,
            cls.Magnet: 7.020376840349667e-06,
            cls.GlassContainer: 7.009144237405107e-06,
            cls.FireAlarm: 6.961967305037957e-06,
            cls.Cradle: 6.94174861973775e-06,
            cls.ChargerAndWire: 6.932762537382102e-06,
            cls.SaltContainer: 6.928269496204279e-06,
            cls.PowerSurge: 6.9215299344375434e-06,
            cls.DecorativeDish: 6.845148234414539e-06,
            cls.Yarmulka: 6.840655193236715e-06,
            cls.ToyCashRegister: 6.8361621520588915e-06,
            cls.Corkscrew: 6.833915631469979e-06,
            cls.ElectricToothbrushBase: 6.818189987347596e-06,
            cls.GlassPot: 6.777752616747182e-06,
            cls.PoolSticks: 6.771013054980446e-06,
            cls.FoodProcessor: 6.753040890269151e-06,
            cls.MotionCamera: 6.73956176673568e-06,
            cls.MugHanger: 6.642961381412469e-06,
            cls.Router: 6.582305325511847e-06,
            cls.Cannister: 6.541867954911433e-06,
            cls.Modem: 6.5306353519668735e-06,
            cls.BoxOfZiplockBags: 6.523895790200138e-06,
            cls.GlassSet: 6.521649269611226e-06,
            cls.MugHolder: 6.485704940188636e-06,
            cls.Whisk: 6.416062801932367e-06,
            cls.BagOfHotDogBuns: 6.407076719576719e-06,
            cls.Vegetable: 6.366639348976306e-06,
            cls.Telescope: 6.350913704853922e-06,
            cls.PepperGrinder: 6.319462416609156e-06,
            cls.Figurine: 6.303736772486772e-06,
            cls.DecorationItem: 6.296997210720037e-06,
            cls.CanOpener: 6.213875948930297e-06,
            cls.BagOfBagels: 6.159959454796412e-06,
            cls.CardboardSheet: 6.126261645962733e-06,
            cls.Flashlight: 6.103796440073614e-06,
            cls.FileStand: 6.090317316540142e-06,
            cls.DrawerHandle: 6.0543729871175525e-06,
            cls.LintRoller: 6.01618213710605e-06,
            cls.Duster: 5.948786519438693e-06,
            cls.SoapBox: 5.928567834138486e-06,
            cls.Duck: 5.910595669427191e-06,
            cls.HandSculpture: 5.910595669427191e-06,
            cls.TreeSculpture: 5.86117221647113e-06,
            cls.ChildCarrier: 5.665724925235795e-06,
            cls.PaperTowelHolder: 5.6477527605245e-06,
            cls.ClothingHanger: 5.643259719346676e-06,
            cls.ClothBag: 5.571371060501495e-06,
            cls.FoodWrappedOnATray: 5.564631498734759e-06,
            cls.OvenMitt: 5.4500589487002534e-06,
            cls.EducationalDisplay: 5.420854181044399e-06,
            cls.Juicer: 5.409621578099839e-06,
            cls.Lid: 5.3826633310328964e-06,
            cls.PieceOfWood: 5.380416810443984e-06,
            cls.Vegetables: 5.378170289855073e-06,
            cls.CanOfFood: 5.362444645732689e-06,
            cls.WoodenKitchenUtensils: 5.355705083965953e-06,
            cls.WineAccessory: 5.353458563377042e-06,
            cls.ChartRoll: 5.348965522199218e-06,
            cls.Balloon: 5.337732919254658e-06,
            cls.ToiletPaperHolder: 5.335486398665747e-06,
            cls.Sock: 5.328746836899011e-06,
            cls.LampShade: 5.261351219231654e-06,
            cls.ToyDoll: 5.25685817805383e-06,
            cls.CellPhoneCharger: 5.250118616287095e-06,
            cls.ToiletBowlBrush: 5.2478720956981824e-06,
            cls.LunchBag: 5.2119277662755925e-06,
            cls.PaperBundle: 5.171490395675178e-06,
            cls.WoodenContainer: 5.155764751552795e-06,
            cls.FilingShelves: 5.153518230963883e-06,
            cls.DecorativeItem: 5.140039107430412e-06,
            cls.HockeyGlove: 5.079383051529791e-06,
            cls.SoccerBall: 5.079383051529791e-06,
            cls.BookHolder: 5.070396969174143e-06,
            cls.ServingPlatter: 5.047931763285024e-06,
            cls.PlasticBowl: 5.032206119162641e-06,
            cls.GlassDish: 4.969303542673108e-06,
            cls.PlasticToyContainer: 4.827772745571659e-06,
            cls.VegetablePeeler: 4.825526224982746e-06,
            cls.Dishes: 4.789581895560156e-06,
            cls.ShowerCap: 4.785088854382333e-06,
            cls.PlasticRack: 4.780595813204509e-06,
            cls.PhoneJack: 4.767116689671038e-06,
            cls.OilContainer: 4.735665401426271e-06,
            cls.LightingTrack: 4.713200195537152e-06,
            cls.Squash: 4.679502386703473e-06,
            cls.CdDisc: 4.672762824936738e-06,
            cls.FlowerBox: 4.6502976190476195e-06,
            cls.VhsTapes: 4.623339371980676e-06,
            cls.Certificate: 4.573915919024615e-06,
            cls.WoodenToy: 4.562683316080055e-06,
            cls.CreamTube: 4.560436795491143e-06,
            cls.HamburgerBun: 4.560436795491143e-06,
            cls.EyeGlasses: 4.5581902749022315e-06,
            cls.Knob: 4.555943754313319e-06,
            cls.Bookrack: 4.524492466068553e-06,
            cls.DoorWayArch: 4.4795620542903155e-06,
            cls.Jug: 4.4728224925235795e-06,
            cls.BusinessCards: 4.42564556015643e-06,
            cls.ToyCylinder: 4.4233990395675175e-06,
            cls.Corn: 4.3941942719116635e-06,
            cls.VideoGame: 4.333538216011042e-06,
            cls.DoorFacingTrimreflection: 4.311073010121923e-06,
            cls.LegOfAGirl: 4.3020869277662755e-06,
            cls.Wii: 4.2773752012882445e-06,
            cls.Rope: 4.2594030365769495e-06,
            cls.ToyApple: 4.252663474810214e-06,
            cls.TrackLight: 4.239184351276742e-06,
            cls.ServingSpoon: 4.178528295376121e-06,
            cls.GiftWrappingRoll: 4.099900074764205e-06,
            cls.Avocado: 4.095407033586382e-06,
            cls.Cake: 4.070695307108351e-06,
            cls.Cleaner: 4.050476621808143e-06,
            cls.Bar: 4.005546210029906e-06,
            cls.DisplayCase: 3.99880664826317e-06,
            cls.Wallet: 3.980834483551875e-06,
            cls.ModelBoat: 3.965108839429491e-06,
            cls.HangingHooks: 3.935904071773637e-06,
            cls.Newspapers: 3.913438865884518e-06,
            cls.ConsoleSystem: 3.864015412928457e-06,
            cls.Tumbler: 3.859522371750633e-06,
            cls.FramedCertificate: 3.852782809983897e-06,
            cls.Button: 3.794373274672188e-06,
            cls.BagOfOreo: 3.7449498217161262e-06,
            cls.Money: 3.7337172187715665e-06,
            cls.Cone: 3.724731136415919e-06,
            cls.Pig: 3.709005492293536e-06,
            cls.CasseroleDish: 3.6955263687600642e-06,
            cls.KichenTowel: 3.6865402864044167e-06,
            cls.HandSanitizerDispenser: 3.684293765815505e-06,
            cls.WallStand: 3.661828559926386e-06,
            cls.FlowerBasket: 3.623637709914884e-06,
            cls.Baseball: 3.5921864216701173e-06,
            cls.PhotoAlbum: 3.5719677363699104e-06,
            cls.ShowerKnob: 3.5719677363699104e-06,
            cls.IncenseCandle: 3.553995571658615e-06,
            cls.FiberglassCase: 3.4888464745801703e-06,
            cls.AluminiumFoil: 3.448409103979756e-06,
            cls.StorageBin: 3.4214508569128133e-06,
            cls.LitterBox: 3.412464774557166e-06,
            cls.ConchShell: 3.3989856510236946e-06,
            cls.ShowerHose: 3.3787669657234873e-06,
            cls.PenBox: 3.3540552392454567e-06,
            cls.TinFoil: 3.342822636300897e-06,
            cls.OrangeJuicer: 3.324850471589602e-06,
            cls.CopperVessel: 3.32260395100069e-06,
            cls.Peach: 3.3158643892339545e-06,
            cls.PepperShaker: 3.223757045088567e-06,
            cls.DecorativeEgg: 3.194552277432712e-06,
            cls.Wreathe: 3.1675940303657697e-06,
            cls.KitchenUtensils: 3.154114906832298e-06,
            cls.Matchbox: 3.138389262709915e-06,
            cls.CleaningWipes: 3.1204170979986195e-06,
            cls.PackageOfWater: 3.113677536231884e-06,
            cls.DishBrush: 3.050774959742351e-06,
            cls.HorseToy: 3.0485284391534392e-06,
            cls.BottleOfKetchup: 3.035049315619968e-06,
            cls.Razor: 3.0193236714975845e-06,
            cls.MakeupBrush: 3.008091068553025e-06,
            cls.ShowerPipe: 2.9743932597193466e-06,
            cls.GlassWare: 2.940695450885668e-06,
            cls.Eraser: 2.9384489302967562e-06,
            cls.SpotLight: 2.9339558891189325e-06,
            cls.IncenseHolder: 2.909244162640902e-06,
            cls.Doily: 2.8890254773406946e-06,
            cls.Key: 2.8732998332183115e-06,
            cls.Cat: 2.8485881067402805e-06,
            cls.WireTray: 2.841848544973545e-06,
            cls.Microphone: 2.819383339084426e-06,
            cls.Vuvuzela: 2.8059042155509545e-06,
            cls.BackScrubber: 2.7766994478951e-06,
            cls.Envelopes: 2.7722064067172765e-06,
            cls.TeaCoaster: 2.7699598861283646e-06,
            cls.PingPongRacquet: 2.7452481596503336e-06,
            cls.ToysBasket: 2.74075511847251e-06,
            cls.HandWeight: 2.7227829537612144e-06,
            cls.Shofar: 2.713796871405567e-06,
            cls.ToyTriangle: 2.711550350816655e-06,
            cls.Trolley: 2.709303830227743e-06,
            cls.FruitPlatter: 2.7025642684610076e-06,
            cls.IronBox: 2.6643734184495053e-06,
            cls.Coins: 2.6598803772716816e-06,
            cls.Cologne: 2.612703444904532e-06,
            cls.Car: 2.608210403726708e-06,
            cls.BicycleHelmet: 2.5812521566597652e-06,
            cls.SixPackOfBeer: 2.565526512537382e-06,
            cls.RolledUpRug: 2.549800868414999e-06,
            cls.ShavingCream: 2.5318287037037036e-06,
            cls.ShowerTube: 2.5183495801702324e-06,
            cls.HoolaHoop: 2.5116100184034968e-06,
            cls.OrangePlasticCap: 2.457693524269611e-06,
            cls.RollOfToiletPaper: 2.4352283183804923e-06,
            cls.SurgeProtect: 2.4105165919024613e-06,
            cls.BottleOfHandWashLiquid: 2.4060235507246375e-06,
            cls.SoapTray: 2.399283988957902e-06,
            cls.Ashtray: 2.3902979066022544e-06,
            cls.Album: 2.3858048654244307e-06,
            cls.Bulb: 2.3858048654244307e-06,
            cls.Tape: 2.376818783068783e-06,
            cls.Grapefruit: 2.2802183977455715e-06,
            cls.Garlic: 2.2173158212560386e-06,
            cls.PenHolder: 2.210576259489303e-06,
            cls.SugerJar: 2.1813714918334482e-06,
            cls.MeasuringCup: 2.1611528065332414e-06,
            cls.ToyPhone: 2.140934121233034e-06,
            cls.ToyRectangle: 2.1297015182884748e-06,
            cls.TissueRoll: 2.1207154359328273e-06,
            cls.Knobs: 2.1184689153439154e-06,
            cls.SculptureOfTheEiffelTower: 2.1139758741660917e-06,
            cls.PersonalCareLiquid: 2.0757850241545894e-06,
            cls.ToyCube: 2.0151289682539685e-06,
            cls.ToyCuboid: 1.9926637623648493e-06,
            cls.DogToy: 1.9904172417759375e-06,
            cls.ClothDryingStand: 1.9746915976535543e-06,
            cls.StackedBinsBoxes: 1.9612124741200827e-06,
            cls.WoodenUtensils: 1.936500747642052e-06,
            cls.Coaster: 1.9185285829307567e-06,
            cls.ToiletPlunger: 1.911789021164021e-06,
            cls.MicrophoneStand: 1.9005564182194617e-06,
            cls.Ruler: 1.8983098976305499e-06,
            cls.CordlessTelephone: 1.896063377041638e-06,
            cls.StackedPlasticRacks: 1.8511329652634001e-06,
            cls.DisplayPlatter: 1.8466399240855762e-06,
            cls.PerfumeBox: 1.8466399240855762e-06,
            cls.Photo: 1.8286677593742812e-06,
            cls.VesselSet: 1.7725047446514838e-06,
            cls.Mailshelf: 1.7365604152288934e-06,
            cls.Toiletries: 1.7028626063952151e-06,
            cls.DecorativeBottle: 1.7006160858063032e-06,
            cls.ToyBottle: 1.7006160858063032e-06,
            cls.HeadPhone: 1.6983695652173914e-06,
            cls.WallHandSanitizerDispenser: 1.6961230446284795e-06,
            cls.PaperWeight: 1.6893834828617437e-06,
            cls.SurgeProtector: 1.6871369622728318e-06,
            cls.Headband: 1.682643921095008e-06,
            cls.CansOfCatFood: 1.6759043593282724e-06,
            cls.SculptureOfTheEmpireStateBuilding: 1.6309739475500346e-06,
            cls.LightBulb: 1.6287274269611227e-06,
            cls.BoxOfPaper: 1.6174948240165631e-06,
            cls.CordlessPhone: 1.6107552622498275e-06,
            cls.SaltAndPepper: 1.5433596445824707e-06,
            cls.ButterflySculpture: 1.5411131239935589e-06,
            cls.Placard: 1.5411131239935589e-06,
            cls.BottleOfComet: 1.5231409592822637e-06,
            cls.Bagel: 1.514154876926616e-06,
            cls.Boomerang: 1.5051687945709685e-06,
            cls.ContactLensSolutionBottle: 1.5006757533931447e-06,
            cls.SoapStand: 1.491689671037497e-06,
            cls.Centerpiece: 1.4647314239705544e-06,
            cls.ToyDog: 1.4624849033816425e-06,
            cls.BottleOfContactLensSolution: 1.4579918622038188e-06,
            cls.TelephoneCord: 1.4557453416149069e-06,
            cls.Ipod: 1.4512523004370831e-06,
            cls.HeatingTray: 1.4445127386703473e-06,
            cls.HandBlender: 1.4400196974925236e-06,
            cls.Hooks: 1.4400196974925236e-06,
            cls.Shovel: 1.433280135725788e-06,
            cls.Handle: 1.4175544916034046e-06,
            cls.TeaCannister: 1.4175544916034046e-06,
            cls.Cane: 1.4018288474810213e-06,
            cls.StackOfPlates: 1.3995823268921095e-06,
            cls.Watch: 1.3995823268921095e-06,
            cls.FlaskSet: 1.38834972394755e-06,
            cls.OvenHandle: 1.3816101621808143e-06,
            cls.Eggs: 1.3793636415919024e-06,
            cls.HandSanitizer: 1.368131038647343e-06,
            cls.KitchenContainerPlastic: 1.368131038647343e-06,
            cls.Stick: 1.3613914768806072e-06,
            cls.UtensilContainer: 1.3434193121693122e-06,
            cls.KitchenUtensil: 1.3389262709914883e-06,
            cls.Torah: 1.327693668046929e-06,
            cls.Hookah: 1.3187075856912814e-06,
            cls.BottleOfPerfume: 1.3164610651023694e-06,
            cls.Drain: 1.3164610651023694e-06,
            cls.WoodenPlank: 1.2895028180354267e-06,
            cls.MiniDisplayPlatform: 1.2602980503795721e-06,
            cls.Kinect: 1.2580515297906603e-06,
            cls.PlasticCupOfCoffee: 1.2221072003680699e-06,
            cls.Mezuza: 1.2153676386013343e-06,
            cls.Earplugs: 1.2108745974235105e-06,
            cls.Inkwell: 1.1636976650563608e-06,
            cls.Notecards: 1.1502185415228894e-06,
            cls.Alarm: 1.1457255003450656e-06,
            cls.ToyPyramid: 1.1457255003450656e-06,
            cls.BottleOfLiquid: 1.1299998562226823e-06,
            cls.Perfume: 1.0895624856222682e-06,
            cls.Bracelet: 1.0626042385553255e-06,
            cls.Sifter: 1.049125115021854e-06,
            cls.PlasticTray: 1.0468785944329422e-06,
            cls.WhiteboardMarker: 1.0176738267770877e-06,
            cls.DecorativeCandle: 1.013180785599264e-06,
            cls.SculptureOfTheChryslerBuilding: 1.0086877444214402e-06,
            cls.UsbDrive: 9.794829767655854e-07,
            cls.Magic8Ball: 9.772364561766735e-07,
            cls.WhiteboardEraser: 9.660038532321142e-07,
            cls.BottleOfSoap: 9.48031688520819e-07,
            cls.ToothpasteHolder: 9.233199620427881e-07,
            cls.ToyPlane: 9.120873590982286e-07,
            cls.Comb: 9.05347797331493e-07,
            cls.IdCard: 8.941151943869335e-07,
            cls.DollarBill: 8.85129112031286e-07,
            cls.Pyramid: 8.604173855532551e-07,
            cls.Walkietalkie: 8.559243443754313e-07,
            cls.Crayon: 8.536778237865194e-07,
            cls.LightSwitchreflection: 8.469382620197837e-07,
            cls.WoodenUtensil: 8.469382620197837e-07,
            cls.ComputerDisk: 8.401987002530481e-07,
            cls.EthernetJack: 8.334591384863124e-07,
            cls.ContainerOfSkinCream: 8.289660973084886e-07,
            cls.YellowPepper: 8.222265355417529e-07,
            cls.Label: 7.885287267080745e-07,
            cls.CanOfBeer: 7.683100414078675e-07,
            cls.Quill: 7.683100414078675e-07,
            cls.Hammer: 7.121470266850701e-07,
            cls.Iphone: 7.031609443294226e-07,
            cls.EyeballPlasticBall: 7.009144237405107e-07,
            cls.PuppyToy: 7.009144237405107e-07,
            cls.GoldPiece: 6.94174861973775e-07,
            cls.Ipad: 6.851887796181275e-07,
            cls.Torch: 6.649700943179204e-07,
            cls.BananaPeel: 6.357653266620657e-07,
            cls.WireBoard: 6.065605590062112e-07,
            cls.Cactus: 6.043140384172993e-07,
            cls.Beeper: 5.908349148838279e-07,
            cls.Orange: 5.773557913503566e-07,
            cls.CardboardTube: 5.616301472279733e-07,
            cls.Mellon: 5.391649413388544e-07,
            cls.ContactLensCase: 5.27932338394295e-07,
            cls.LidOfJar: 4.964810501495284e-07,
            cls.Stamp: 4.85248447204969e-07,
            cls.CoffeePot: 4.830019266160571e-07,
            cls.Eggplant: 4.807554060271451e-07,
            cls.Lock: 4.7176932367149757e-07,
            cls.MortarAndPestle: 4.672762824936738e-07,
            cls.Vasoline: 4.672762824936738e-07,
            cls.TeaBox: 4.4930411778237864e-07,
            cls.MedicineTube: 4.2009935012652405e-07,
            cls.CableModem: 4.178528295376121e-07,
            cls.Pepper: 4.133597883597884e-07,
            cls.Fork: 4.0662022659305266e-07,
            cls.StapleRemover: 3.8190850011502185e-07,
            cls.PoolBall: 3.594432942259029e-07,
            cls.Charger: 3.4821069128134346e-07,
            cls.OrnamentalPot: 3.4821069128134346e-07,
            cls.Chapstick: 3.4596417069243154e-07,
            cls.DrawerKnob: 3.414711295146078e-07,
            cls.Nailclipper: 3.1900592362548886e-07,
            cls.FloorTrim: 3.0552680009201747e-07,
            cls.Webcam: 3.032802795031056e-07,
            cls.Utensil: 2.987872383252818e-07,
            cls.DoorLock: 2.8081507361398667e-07,
            cls.PingPongRacket: 2.8081507361398667e-07,
            cls.LintComb: 2.673359500805153e-07,
            cls.Kiwi: 2.4936378536922015e-07,
            cls.SecurityCamera: 2.471172647803083e-07,
            cls.RollOfPaperTowels: 2.1341945594662986e-07,
            cls.Wine: 2.066798941798942e-07,
            cls.PingPongBall: 1.8421468829077526e-07,
            cls.Trinket: 1.662425235794801e-07,
            cls.Lemon: 1.5950296181274443e-07,
        }
