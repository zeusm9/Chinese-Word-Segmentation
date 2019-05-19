# Generates a file without white spaces and a file with characters in 'BIES' removing empty lines from the original file
def generate_files(input_path, input_name, label_name):
    input_file = open(input_name, 'w',encoding='utf8')
    label_file = open(label_name, 'w',encoding='utf8')
    original_data = open(input_path, 'r',encoding='utf8')

    for line in original_data.readlines():
        if not line.isspace():
            words = line.strip().split()
            for word in words:
                word.replace(' ', '')
                input_file.write(word)
                if len(word) == 1:
                    label_file.write('S')
                else:
                    label_file.write('B')
                    for char in word[1:len(word) - 1]:
                        label_file.write('I')
                    label_file.write('E')
            label_file.write("\n")
            input_file.write("\n")

    input_file.close()
    label_file.close()
    original_data.close()


# Read data from file and put them in a list of lists where each list represents a line of text in the original file
def read_data(path):
    data = []
    with open(path,encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            paragraph = []
            for unigram in line:
                if unigram != '\n':
                    paragraph.append(unigram)
            data.append(paragraph)
    return data
