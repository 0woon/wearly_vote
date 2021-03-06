# Generated by Django 3.1 on 2020-08-31 08:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wearly', '0002_auto_20200826_1516'),
    ]

    operations = [
        migrations.CreateModel(
            name='wear',
            fields=[
                ('idx', models.BigAutoField(primary_key=True, serialize=False)),
                ('image_id', models.IntegerField()),
                ('post_id', models.TextField()),
                ('image_file_name', models.TextField()),
                ('hashtag_crawl', models.TextField()),
                ('like_num', models.IntegerField()),
                ('comment_num', models.IntegerField()),
                ('account_name', models.TextField(blank=True)),
                ('fashion', models.IntegerField()),
                ('ootd', models.IntegerField()),
                ('fashionblogger', models.IntegerField()),
                ('instafashion', models.IntegerField()),
                ('fashionista', models.IntegerField()),
                ('streetstyle', models.IntegerField()),
                ('outfit', models.IntegerField()),
                ('instagood', models.IntegerField()),
                ('fashionable', models.IntegerField()),
                ('fashionstyle', models.IntegerField()),
                ('stylish', models.IntegerField()),
                ('outfitoftheday', models.IntegerField()),
                ('styleblogger', models.IntegerField()),
                ('moda', models.IntegerField()),
                ('love', models.IntegerField()),
                ('model', models.IntegerField()),
                ('look', models.IntegerField()),
                ('streetwear', models.IntegerField()),
                ('photooftheday', models.IntegerField()),
                ('streetfashion', models.IntegerField()),
                ('instastyle', models.IntegerField()),
                ('fashionweek', models.IntegerField()),
                ('photography', models.IntegerField()),
                ('trend', models.IntegerField()),
                ('fashiongram', models.IntegerField()),
                ('beautiful', models.IntegerField()),
                ('fashionblog', models.IntegerField()),
                ('fashionaddict', models.IntegerField()),
                ('beauty', models.IntegerField()),
                ('summer', models.IntegerField()),
                ('fashiondiaries', models.IntegerField()),
                ('fashionpost', models.IntegerField()),
                ('fashioninspiration', models.IntegerField()),
                ('lookoftheday', models.IntegerField()),
                ('dress', models.IntegerField()),
                ('blogger', models.IntegerField()),
                ('picoftheday', models.IntegerField()),
                ('lookbook', models.IntegerField()),
                ('girl', models.IntegerField()),
                ('mensfashion', models.IntegerField()),
                ('cute', models.IntegerField()),
                ('follow', models.IntegerField()),
                ('instagram', models.IntegerField()),
                ('fashionshow', models.IntegerField()),
                ('lifestyle', models.IntegerField()),
                ('shopping', models.IntegerField()),
                ('fashioninsta', models.IntegerField()),
                ('dailylook', models.IntegerField()),
                ('sporty', models.IntegerField()),
                ('casual', models.IntegerField()),
                ('modern', models.IntegerField()),
                ('elegant', models.IntegerField()),
                ('natural', models.IntegerField()),
                ('glamorous', models.IntegerField()),
                ('sophisticated', models.IntegerField()),
                ('grunge', models.IntegerField()),
                ('retro', models.IntegerField()),
                ('romantic', models.IntegerField()),
                ('sexy', models.IntegerField()),
                ('military', models.IntegerField()),
                ('ethnic', models.IntegerField()),
                ('classic', models.IntegerField()),
                ('business_casual', models.IntegerField()),
                ('manish', models.IntegerField()),
                ('exotic', models.IntegerField()),
                ('goth_punk_rocker', models.IntegerField()),
                ('hiphop', models.IntegerField()),
                ('hippie', models.IntegerField()),
                ('tomboy', models.IntegerField()),
                ('preppy', models.IntegerField()),
                ('kitsch_kidult', models.IntegerField()),
            ],
        ),
    ]
