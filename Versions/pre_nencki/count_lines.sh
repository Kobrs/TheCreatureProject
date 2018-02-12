# POSIX

#this example prints line count for all found files
total='0'
{
find . -type f -name "*.py" | while read FILE; do
    #you see use grep instead wc ! for properly counting
    count=$(grep -c ^ < "$FILE")
    echo "$FILE has $count lines"
    let total+=count #in bash, you can convert this for another shell

echo $total
done
# echo TOTAL LINES COUNTED:  ${total}
echo ${total}
}