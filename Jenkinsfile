def builders = [:]

builders['master'] = {
  node('master') {
    checkout scm
    sh 'echo "On a Mac"'
  }
}

builders['benchpress'] = {
  node('benchpress') {
    checkout scm
    sh 'echo "On a Linux"'
  }
}

parallel builders
