#!/bin/bash
mkdir -p linux_practice/name
mkdir -p linux_practice/permission
touch linux_practice/name/file1.txt
touch linux_practice/name/file2.txt
touch linux_practice/permission/file3.txt
touch linux_practice/permission/file4.txt
rm -f linux_practice/name/file1.txt
mv linux_practice/name/file2.txt linux_practice/name/show.txt
echo "Hello linux" > linux_practice/name/show.txt
cat linux_practice/name/show.txt
for file in linux_practice/permission/file3.txt linux_practice/permission/file4.txt; do
    chmod 644 "$file"
    echo "Changed permissions for $(basename "$file") to -rw-r--r--"
done
