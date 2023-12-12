from enum import IntEnum, unique


@unique
class Components(IntEnum):

    # networks
    Encoder = 0
    DecodingGenerator = 1
    Discriminator = 2
    Conditional = 3

    # other entities
    OriginalCodecData = 4
    ReconstructedCodecData = 5
    CompleteData = 6
    AuxillaryOutput = 7
