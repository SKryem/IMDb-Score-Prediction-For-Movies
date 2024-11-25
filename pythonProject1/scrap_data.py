from bs4 import BeautifulSoup
import requests
from utils import get_movie_id, write_rows_to_csv, get_movie_links, load_checkpoint, save_checkpoint, headers
import atexit

# Get the text from imdb  of the first 10(or less) reviews

reviews_count_checkpoint = 0


def get_and_write_num_of_reviews_aux(session, movie_links: list, offset=0):
    global reviews_count_checkpoint
    reviews_count_checkpoint = offset
    for index in range(len(movie_links)):
        if index < offset:
            continue
        movie_link = movie_links[index]
        number_of_reviews = 0
        response = session.get(movie_link, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        movie_title = soup.find('span', {'data-testid': 'hero__primary-text'})
        scores = soup.find_all('span', {'class': 'score'})  # returns all scores in page
        for score in scores:
            if score.findNext().text == "Critic reviews":  # get only num of critic reviews
                number_of_reviews = score.text
                continue

        if movie_title is None:
            continue
        movie_title = movie_title.text
        number_of_reviews = int(number_of_reviews)

        data = {'movie_title': movie_title, 'Num Of Reviews': number_of_reviews}
        write_rows_to_csv('./data/critic_reviews_number.csv', data, ['movie_title', "Num Of Reviews"])
        reviews_count_checkpoint = index + 1

    return


def scrap_reviews_count():
    print("Scrapping Reviews Count Start")
    offset = load_checkpoint("./checkpoints/reviews_count_checkpoint.txt")  # number of movies already done
    print(f'Starting from checkpoint: {offset}')
    with requests.session() as session:
        get_and_write_num_of_reviews_aux(session, get_movie_links(), offset)
    print("Scrapping Reviews Count Done")

    atexit.register(save_checkpoint, "./checkpoints/reviews_count_checkpoint.txt", reviews_count_checkpoint)


reviews_checkpoint = 0  # used to continue from where the last run stopped


def get_and_write_all_critic_reviews_aux(session, movie_links, offset=0):
    global reviews_checkpoint
    reviews_checkpoint = offset
    for index in range(len(movie_links)):
        if index < offset:
            continue
        data = list()
        movie_link = movie_links[index]
        movie_id = get_movie_id(movie_link)
        reviews_link = f"https://www.imdb.com/title/{movie_id}/criticreviews/?ref_=tt_ov_rt"
        response = session.get(reviews_link, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        #movie_title = soup.find('h2', {'class': "sc-5f0a904b-9 bMRMHO"}).text # imdb site changed class name..
        movie_title = soup.find('h2', {'class': "sc-b8cc654b-9 dmvgRY"}).text
        first_texts_container = soup.find_all('span', {'class': "sc-d8486f96-6 bwEuBg"})
        second_texts_container = soup.find_all('a', {'class': "ipc-link ipc-link--base sc-d8486f96-6 bwEuBg"})
        reviews_container = list()

        for par in first_texts_container:
            reviews_container.append(par.next.next.text)
        for par in second_texts_container:
            reviews_container.append(par.next.next.next.next.text)
        del (reviews_container[0::2])  # due to the way the page of IMDb is formatted

        for review in reviews_container:
            data.append({'movie_title': movie_title, 'Review': review})
        write_rows_to_csv('./data/imdb_reviews.csv', data, ['movie_title', "Review"])
        reviews_checkpoint = index + 1
    return


def scrap_critic_reviews():
    print("Scrapping Critic Reviews Start")
    offset = load_checkpoint('./checkpoints/reviews_checkpoint.txt')
    print(f'Starting from checkpoint: {offset}')
    with requests.session() as session:
        get_and_write_all_critic_reviews_aux(session, get_movie_links(), offset)
    print("Scrapping Critic Reviews Done ")

    atexit.register(save_checkpoint, './checkpoints/reviews_checkpoint.txt', reviews_checkpoint)


def scrap_data():
    print("****************************")
    print("Scrapping")
    print("----")
    scrap_critic_reviews()
    print("----")
    scrap_reviews_count()

# test code to scrap reviews of 1 movie (just change the tt... in the datum link):


# with requests.session() as session:
#     datum = "https://www.imdb.com/title/tt2379713/criticreviews/?ref_=tt_ov_rt"
#     response = session.get(datum, headers=headers)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     movie_title = soup.find('h2', {'class': "sc-5f0a904b-9 bMRMHO"}).text
#     first_texts_container = soup.find_all('span', {'class': "sc-d8486f96-6 bwEuBg"})
#     second_texts_container = soup.find_all('a', {'class': "ipc-link ipc-link--base sc-d8486f96-6 bwEuBg"})
#
#     reviews_container = list()
#     for text in first_texts_container:
#         reviews_container.append(text.next.next.text)
#
#     for text in second_texts_container:
#         reviews_container.append(text.next.next.next.next.text)
#
#     del (reviews_container[0::2])  # due to the way the page of IMDb is formatted
#     # we get some unwanted "texts", the relevant reviews are in the odd indices,
#     #                             # so we remove the rest
#     for i in range(len(reviews_container):
#         print((reviews_container[i]) + '\n')
