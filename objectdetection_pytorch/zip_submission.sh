while getopts p: flag
do
    case "${flag}" in
        p) path=${OPTARG};;
        esac
done

dir_count=$(ls center_logs | wc -l)

zip -r ex9_mlrcv_submission.zip mlrcv/*.py center_logs/centernet_$((dir_count-1))/ ex9_object_detection.ipynb