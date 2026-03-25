SELECT
    f.repo_name,
    f.ref,
    f.path,
    c.copies,
    c.content
FROM `bigquery-public-data.github_repos.files` as f
         JOIN `bigquery-public-data.github_repos.contents` as c
              on f.id = c.id
WHERE
    NOT c.binary
    AND f.path like '%.c'
    OR f.path like '%.h'