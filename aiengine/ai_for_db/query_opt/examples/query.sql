SELECT MIN(t.title) AS title
FROM title AS t
JOIN movie_keyword AS mk ON mk.movie_id = t.id
JOIN keyword AS k ON k.id = mk.keyword_id
WHERE k.keyword = 'computer-animation'
  AND t.title = 'Shrek 2'
  AND t.production_year BETWEEN 2000 AND 2010;
