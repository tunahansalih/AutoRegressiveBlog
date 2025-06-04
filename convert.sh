nb=$1

function convert(){
	jupyter nbconvert --to markdown $nb
	python edit.py ${nb%.ipynb}.md
	mv ${nb%.ipynb}.md post
	mv ${nb%.ipynb}_files images
	echo "==========Conversion complete!=========="
}

convert