find . -type f -name "*inv_pendulum*" -exec sh -c '
    for file do
        newname=$(echo "$file" | sed "s/inv_pendulum/pendulum/g")
        if [ "$file" != "$newname" ]; then
            mv "$file" "$newname"
            echo "Renamed $file to $newname"
        fi
    done
' sh {} +
