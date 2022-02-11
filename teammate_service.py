import time


def main():
    image = 'Snapchat_logo.svg.png'

    while True:
        try:
            with open('request.txt') as file:
                contents = file.read()
                break
        except FileNotFoundError:
            print("Waiting...")
            time.sleep(2)

    if contents == 'Snap':
        with open('response.txt', 'w') as file:
            file.write(image)


if __name__ == '__main__':
    main()
