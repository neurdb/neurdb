# from peewee import *

# db = SqliteDatabase(None)


# class Stats(Model):
#     iteration = IntegerField(null=True)
#     notebook = CharField()
#     dataset = CharField()
#     hipipe_acc = FloatField(null=True)
#     aipipe_acc = FloatField()
#     haipipe_acc = FloatField(null=True)
#     execution_time = FloatField(null=True)

#     class Meta:
#         database = db


# def init_stats_db(file_path: str):
#     global db
#     db.init(file_path)
#     db.connect()
#     db.create_tables([Stats])
