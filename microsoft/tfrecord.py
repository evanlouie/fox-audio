from typing import List
import argparse
import tensorflow as tf


def read(record_path: str, is_sequence_example: bool = True) -> List:
    """
    Return the contents of a TF Record
    Note: Tensorflow uses weird reflection on protobufs. The FromString method does exist; disable E1101 warning
    """
    return [
        (
            tf.train.SequenceExample.FromString(record)  # pylint: disable=E1101
            if is_sequence_example
            else tf.train.Example.FromString(record)  # pylint: disable=E1101
        )
        for record in tf.python_io.tf_record_iterator(record_path)
    ]


def print_example(record_path: str, is_sequence_example: bool = True) -> None:
    """Prints the contents of a TF Record file containing either a SequenceExample or Example."""
    examples = read(record_path, is_sequence_example)
    for example in examples:
        print(example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_example", help="target TFRecord SequenceExample file")
    parser.add_argument("--example", help="target TFRecord Example file")
    args = parser.parse_args()

    if args.example:
        print_example(args.example, is_sequence_example=False)
    else:
        print_example(args.sequence_example)
