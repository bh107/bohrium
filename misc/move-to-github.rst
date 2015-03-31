==============
Move to github
==============

Clone bitbucket repos::

    git clone git@bitbucket.org:bohrium/bohrium.git

Fetch all branches::

    #!/bin/bash
    for branch in `git branch -a | grep remotes | grep -v HEAD | grep -v master`; do
        git branch --track ${branch##*/} $branch
    done


Fix authors::

    git filter-branch --env-filter '

        troels="committer*"
        mads="Mads Kristensen*"
        kenneth="kenkendk*"
        simon1="safl*"
        simon2="Simon Andreas*"
        andreas1="andreas*"
        andreas2="Andreas*"
        case "$GIT_COMMITTER_NAME" in
            $troels) touch "$GIT_COMMITTER_NAME"
                  GIT_COMMITTER_NAME="Troels Blum";
                  GIT_AUTHOR_NAME="Troels Blum";
                  GIT_COMMITTER_EMAIL="troels@blum.dk";
                  GIT_AUTHOR_EMAIL="troels@blum.dk";;

            $mads) touch "$GIT_COMMITTER_NAME"
                  GIT_COMMITTER_NAME="Mads R. B. Kristensen";
                  GIT_AUTHOR_NAME="Mads R. B. Kristensen";
                  GIT_COMMITTER_EMAIL="madsbk@gmail.com";
                  GIT_AUTHOR_EMAIL="madsbk@gmail.com";;

            $kenneth) touch "$GIT_COMMITTER_NAME"
                  GIT_COMMITTER_NAME="Kenneth Skovhede";
                  GIT_AUTHOR_NAME="Kenneth Skovhede";;

            $simon1) touch "$GIT_COMMITTER_NAME"
                  GIT_COMMITTER_NAME="Simon A. F. Lund";
                  GIT_AUTHOR_NAME="Simon A. F. Lund";
                  GIT_COMMITTER_EMAIL="safl@safl.dk";
                  GIT_AUTHOR_EMAIL="safl@safl.dk";;

            $simon2) touch "$GIT_COMMITTER_NAME"
                  GIT_COMMITTER_NAME="Simon A. F. Lund";
                  GIT_AUTHOR_NAME="Simon A. F. Lund";
                  GIT_COMMITTER_EMAIL="safl@safl.dk";
                  GIT_AUTHOR_EMAIL="safl@safl.dk";;

            $andreas1) touch "$GIT_COMMITTER_NAME"
                  GIT_COMMITTER_NAME="Andreas Thorning";
                  GIT_AUTHOR_NAME="Andreas Thorning";
                  GIT_COMMITTER_EMAIL="ifcn913@alumni.ku.dk";
                  GIT_AUTHOR_EMAIL="fcn913@alumni.ku.dk";;

            $andreas2) touch "$GIT_COMMITTER_NAME"
                  GIT_COMMITTER_NAME="Andreas Thorning";
                  GIT_AUTHOR_NAME="Andreas Thorning";
                  GIT_COMMITTER_EMAIL="ifcn913@alumni.ku.dk";
                  GIT_AUTHOR_EMAIL="fcn913@alumni.ku.dk";;
        esac

    ' --tag-name-filter cat -- --branches --tags

Fix large core dump file::

    git filter-branch -f --index-filter 'git rm -f --cached --ignore-unmatch "misc/tools/core"' --tag-name-filter cat -- --branches --tags

Now we are ready to push the new repos to github::

    git remote rename origin bitbucket
    git remote add origin git@github.com:bh107/bohrium.git
    git push --force --tags origin 'refs/heads/*'

