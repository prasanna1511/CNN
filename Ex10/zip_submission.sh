while getopts p: flag
do
    case "${flag}" in
        p) path=${OPTARG};;
        esac
done

dir_count=$(ls train_logs | wc -l)

zip -r ex10_mlrcv_submission.zip mlrcv/*.py train_logs/no_transfer_learning/ train_logs/transfer_learning_train_finetune/ train_logs/transfer_learning_train_freezed ex10_transfer_learning.ipynb