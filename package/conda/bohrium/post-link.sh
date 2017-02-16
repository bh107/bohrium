#!/bin/bash

echo "**IMPORTANT** Bohrium uses the config at $PREFIX/etc/bohrium/config.ini" >> $PREFIX/.messages.txt

mkdir -p $PREFIX/etc/conda/deactivate.d
mkdir -p $PREFIX/etc/conda/activate.d
cat <<EOF > $PREFIX/etc/conda/deactivate.d/bohrium.sh
#!/bin/sh
echo unsetting BH_CONFIG
unset BH_CONFIG
EOF
cat <<EOF > $PREFIX/etc/conda/activate.d/bohrium.sh
#!/bin/sh
echo setting BH_CONFIG to $PREFIX/etc/bohrium/config.ini
export BH_CONFIG=$PREFIX/etc/bohrium/config.ini
EOF
