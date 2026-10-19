from datetime import datetime

from sqlmodel import Field, SQLModel

# Note: We only define the fields we strictly need to save memory.


class CardIdentifiers(SQLModel, table=True):
    # __tablename__ = "cardIdentifiers" XX
    uuid: str | None = Field(primary_key=True)
    mtgoId: str | None = Field(default=None)
    scryfallId: str | None = Field(default=None)
    # cardKingdomEtchedId: str | None = Field(default=None)
    # cardKingdomFoilId: str | None = Field(default=None)
    # cardKingdomId: str | None = Field(default=None)
    # cardsphereFoilId: str | None = Field(default=None)
    # cardsphereId: str | None = Field(default=None)
    # deckboxId: str | None = Field(default=None)
    # mcmId: str | None = Field(default=None)
    # mcmMetaId: str | None = Field(default=None)
    # mtgArenaId: str | None = Field(default=None)
    # mtgjsonFoilVersionId: str | None = Field(default=None)
    # mtgjsonNonFoilVersionId: str | None = Field(default=None)
    # mtgjsonV4Id: str | None = Field(default=None)
    # mtgoFoilId: str | None = Field(default=None)
    # multiverseId: str | None = Field(default=None)
    # scryfallCardBackId: str | None = Field(default=None)
    # scryfallIllustrationId: str | None = Field(default=None)
    # scryfallOracleId: str | None = Field(default=None)
    # tcgplayerEtchedProductId: str | None = Field(default=None)
    # tcgplayerProductId: str | None = Field(default=None)


class Cards(SQLModel, table=True):
    # __tablename__ = "cards" XX
    uuid: str | None = Field(primary_key=True)
    name: str | None = Field(default=None)
    manaCost: str | None = Field(default=None)
    setCode: str | None = Field(default=None)
    artist: str | None = Field(default=None)
    number: str | None = Field(default=None)
    type: str | None = Field(default=None)
    power: str | None = Field(default=None)
    toughness: str | None = Field(default=None)
    borderColor: str | None = Field(default=None)
    flavorText: str | None = Field(default=None)  # XX
    language: str | None = Field(default=None)
    text: str | None = Field(default=None)
    frameVersion: str | None = Field(default=None)
    # artistIds: str | None = Field(default=None)
    # asciiName: str | None = Field(default=None)
    # attractionLights: str | None = Field(default=None)
    # availability: str | None = Field(default=None)
    # boosterTypes: str | None = Field(default=None)
    # cardParts: str | None = Field(default=None)
    # colorIdentity: str | None = Field(default=None)
    # colorIndicator: str | None = Field(default=None)
    # colors: str | None = Field(default=None)
    # defense: str | None = Field(default=None)
    # duelDeck: str | None = Field(default=None)
    # edhrecRank: int
    # edhrecSaltiness: floaf | None = Field(default=None)
    # faceConvertedManaCost: float | None = Field(default=None)
    # faceFlavorName: str | None = Field(default=None)
    # faceManaValue: floaf | None = Field(default=None)
    # faceName: str | None = Field(default=None)
    # facePrintedName: str | None = Field(default=None)
    # finishes: str | None = Field(default=None)
    # flavorName: str | None = Field(default=None)
    # frameEffects: str | None = Field(default=None)
    # hand: str | None = Field(default=None)
    # hasAlternativeDeckLimit: bool | None = Field(default=None)
    # hasContentWarning: bool | None = Field(default=None)
    # hasFoil: bool | None = Field(default=None)
    # hasNonFoil: bool | None = Field(default=None)
    # isAlternative: bool | None = Field(default=None)
    # isFullArt: bool | None = Field(default=None)
    # isFunny: bool | None = Field(default=None)
    # isGameChanger: bool | None = Field(default=None)
    # isOnlineOnly: bool | None = Field(default=None)
    # isOversized: bool | None = Field(default=None)
    # isPromo: bool | None = Field(default=None)
    # isRebalanced: bool | None = Field(default=None)
    # isReprint: bool | None = Field(default=None)
    # isReserved: bool | None = Field(default=None)
    # isStarter: bool | None = Field(default=None)
    # isStorySpotlight: bool | None = Field(default=None)
    # isTextless: bool | None = Field(default=None)
    # isTimeshifted: bool | None = Field(default=None)
    # keywords: str | None = Field(default=None)
    # layout: str | None = Field(default=None)
    # leadershipSkills: str | None = Field(default=None)
    # life: str | None = Field(default=None)
    # loyalty: str | None = Field(default=None)
    # manaValue: floaf | None = Field(default=None)
    # originalPrintings: str | None = Field(default=None)
    # originalReleaseDate: str | None = Field(default=None)
    # originalText: str | None = Field(default=None)
    # otherFaceIds: str | None = Field(default=None)
    # printedName: str | None = Field(default=None)
    # printedText: str | None = Field(default=None)
    # printedType: str | None = Field(default=None)
    # printings: str | None = Field(default=None)
    # promoTypes: str | None = Field(default=None)
    # rarity: str | None = Field(default=None)
    # rebalancedPrintings: str | None = Field(default=None)
    # relatedCards: str | None = Field(default=None)
    # securityStamp: str | None = Field(default=None)
    # side: str | None = Field(default=None)
    # signature: str | None = Field(default=None)
    # sourceProducts: str | None = Field(default=None)
    # subsets: str | None = Field(default=None)
    # subtypes: str | None = Field(default=None)
    # supertypes: str | None = Field(default=None)
    # types: str | None = Field(default=None)
    # variations: str | None = Field(default=None)
    # watermark: str | None = Field(default=None)


class Sets(SQLModel, table=True):
    # __tablename__ = "sets"
    code: str = Field(
        primary_key=True,
        index=True,
        max_length=8,
        description="Unique set code",
    )
    name: str | None = Field(None, description="Set name")
    baseSetSize: int | None = Field(default=None, description="Base size of the set")
    releaseDate: str | None = Field(None, description="Release date as string")
    # block: Optional[str] = Field(None, description="Block name the set belongs to")
    # cardsphereSetId: Optional[int] = Field(
    #     None, description="External cardsphere set id"
    # )
    # isFoilOnly: Optional[bool] = Field(None, description="Foil-only flag")
    # isForeignOnly: Optional[bool] = Field(None, description="Foreign-only flag")
    # isNonFoilOnly: Optional[bool] = Field(None, description="Non-foil-only flag")
    # isOnlineOnly: Optional[bool] = Field(None, description="Online-only flag")
    # isPartialPreview: Optional[bool] = Field(None, description="Partial preview flag")
    # keyruneCode: Optional[str] = Field(None, description="Keyrune code")
    # languages: Optional[str] = Field(None, description="Languages for translations")
    # mcmId: Optional[int] = Field(None, description="MCM id")
    # mcmIdExtras: Optional[int] = Field(None, description="Additional MCM id info")
    # mcmName: Optional[str] = Field(None, description="MCM name")
    # mtgoCode: Optional[str] = Field(None, description="MTGO code")
    # parentCode: Optional[str] = Field(None, description="Parent set code")
    # tcgplayerGroupId: Optional[int] = Field(None, description="TCGPlayer group id")
    # tokenSetCode: Optional[str] = Field(None, description="Token set code")
    # totalSetSize: Optional[int] = Field(None, description="Total set size")
    # type: Optional[str] = Field(None, description="Set type")


class Meta(SQLModel, table=True):
    # __tablename__ = "meta" XX
    version: str | None = Field(primary_key=True)
    date: datetime | None = Field(default=None)


class MobileCache(
    SQLModel,
    table=True,
):
    # https://docs.sqlalchemy.org/en/13/orm/extensions/declarative/table_config.html
    __table_args__ = {
        "extend_existing": True,
    }
    # __table_args__ = (
    #     {"extend_existing": True},  # Table Options
    #     Index("number", "baseSetSize"),  # Composite Index
    # )
    uuid: str = Field(primary_key=True, index=True)
    name: str = Field(default=None, index=False)
    setCode: str = Field(default=None, index=False)  # index=True

    number: str = Field(default=None, index=False)
    baseSetSize: str = Field(default=None, index=False)

    # number: str = Field(default=None) # Use with composite index
    # baseSetSize: str = Field(default=None) # Use with composite index

    setReleaseYear: str | None = Field(None, index=False)  # Copyright year
    artist: str | None = Field(default=None)
    url: str | None = Field(default=None)
    manaCost: str | None = Field(default=None)
    type: str | None = Field(default=None)
    # setName: Optional[str] = Field(None, description="Set name")
    power: str | None = Field(default=None)
    toughness: str | None = Field(default=None)
    borderColor: str | None = Field(default=None)
    # text: str | None = Field(default=None)
    # flavorText: str | None = Field(default=None)
    # scryfallId: str
    # mtgoId: str | None = Field(default=None)
    # language: str | None = Field(default=None)
    # frameVersion: str | None = Field(default=None) # removed, useless

    # # AllPrintings, table: cards
    # uuid: str | None = Field(primary_key=True)
    # name: str | None = Field(default=None)
    # manaCost: str | None = Field(default=None)
    # setCode: str | None = Field(default=None)
    # artist: str | None = Field(default=None)
    # number: str | None = Field(default=None)
    # type: str | None = Field(default=None)
    # power: str | None = Field(default=None)
    # toughness: str | None = Field(default=None)
    # borderColor: str | None = Field(default=None)
    # flavorText: str | None = Field(default=None)  # XX
    # language: str | None = Field(default=None)
    # text: str | None = Field(default=None)
    # frameVersion: str | None = Field(default=None)

    # # AllPrintings, table: sets
    # code: str = Field(
    #     primary_key=True,
    #     index=True,
    #     max_length=8,
    #     description="Unique set code",
    # ) # setCode
    # name: Optional[str] = Field(None, description="Set name") # setName
    # baseSetSize: Optional[int] = Field(
    #     default=None, description="Base size of the set"
    # )
    # setReleaseDate
    # releaseDate: Optional[str] = Field(None, description="Release date as string")

    # # AllPrintings, table: cardIdentifiers
    # uuid: str | None = Field(primary_key=True)
    # mtgoId: str | None = Field(default=None)
    # scryfallId: str | None = Field(default=None)


# def main():
#     db_path = "AllPrintings.sqlite"
#     engine = create_engine(f"sqlite:///{db_path}", echo=False)

#     with Session(engine) as session:
#         statement = select(CardIdentifiers).where(
#             CardIdentifiers.uuid == "5f8287b1-5bb6-5f4c-ad17-316a40d5bb0c"
#         )
#         result = session.exec(statement).first()
#         if result is not None:
#             print(f"\nCARD NAME:\n{result.mtgoId}\n")


# if __name__ == "__main__":
#     main()
