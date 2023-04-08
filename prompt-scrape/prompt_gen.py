import itertools
import numpy as np
from nomic import atlas
from tqdm import tqdm
import jsonlines
import random

PERSONS = ['Donald Trump', 'Elon Musk', 'Hamlet', 'Michael Jackson', 'Albert Einstein', 'Cleopatra', 'Oprah Winfrey', 'Beyoncé', 'Taylor Swift', 'Serena Williams', 'Barack Obama', 'Malala Yousafzai', 'Bill Gates', 'Kobe Bryant'] + ['Oprah Winfrey', 'Albert Einstein', 'Cleopatra', 'William Shakespeare', 'Frida Kahlo', 'Isaac Newton', 'Nelson Mandela', 'Jane Austen', 'Sigmund Freud', 'Gandhi', 'Ada Lovelace', 'Leonardo da Vinci', 'Marie Curie', 'Marilyn Monroe', 'Ludwig van Beethoven', 'Joan of Arc', 'Winston Churchill', 'Beyoncé', 'Pablo Picasso', 'Abraham Lincoln', 'Napoleon Bonaparte', 'Galileo Galilei', 'Michael Jordan', 'Sherlock Holmes', 'Robin Hood', 'Jules Verne', 'Homer', 'Mozart', 'Vincent van Gogh', 'Florence Nightingale', 'Harriet Tubman', 'Thomas Edison', 'Confucius', 'Queen Victoria', 'King Arthur', 'Amelia Earhart', 'Plato', 'Malala Yousafzai', 'Nostradamus', 'Alexander the Great', 'Rosa Parks', 'Charlie Chaplin', 'Marlon Brando', 'Charles Darwin', 'Steve Jobs', 'Anne Frank', 'Barack Obama', 'Mother Teresa', 'Vladimir Putin', 'Christopher Columbus', 'Bruce Lee', 'Julius Caesar', 'Edgar Allan Poe', 'Coco Chanel', 'Genghis Khan', 'George Washington', 'Salvador Dalí', 'Helen Keller', 'Audrey Hepburn', 'Alfred Hitchcock', 'Andy Warhol', 'John F. Kennedy', 'Martin Luther King Jr.', 'Indira Gandhi', 'Queen Elizabeth II', 'Bill Gates', 'Benjamin Franklin', 'Jacques Cousteau', 'Frank Sinatra', 'Marilyn Manson', 'Kurt Cobain', 'Simone de Beauvoir', 'Virginia Woolf', 'Marie Antoinette', 'Dalai Lama', 'John Lennon', 'Socrates', 'George Lucas', 'David Bowie', 'Bob Dylan', 'Che Guevara', 'Mata Hari', 'Michael Jackson', 'Oscar Wilde', 'Machiavelli', 'Madonna', 'Johnny Depp', 'Grace Kelly', 'Jimi Hendrix', 'Fidel Castro', 'Neil Armstrong', 'Marie-Antoine Carême', 'Orson Welles', 'Hillary Clinton', 'J.R.R. Tolkien', 'John Steinbeck', 'Edvard Munch', 'Ernest Hemingway', 'F. Scott Fitzgerald', 'Tennessee Williams', 'Aristotle', 'Sun Tzu', 'Immanuel Kant', 'Carl Jung', 'Jean-Paul Sartre', 'Susan B. Anthony', 'Gloria Steinem', 'Ruth Bader Ginsburg', 'Angela Merkel', 'Rumi', 'Laozi', 'Emily Dickinson', 'Mark Twain', 'Nikola Tesla', 'Carl Sagan', 'J.K. Rowling', 'Alan Turing', 'H.G. Wells', 'Ray Bradbury', 'Jane Goodall', 'Chinua Achebe', 'Toni Morrison', 'Maya Angelou', 'Greta Thunberg', 'Fridtjof Nansen', 'Fernando Pessoa', 'Sophocles', 'Eleanor Roosevelt', 'Dante Alighieri', 'Muhammad Ali', 'Charles Lindbergh', 'George Harrison', 'Paul McCartney', 'Ringo Starr', 'Tom Hanks', 'Meryl Streep', 'Nina Simone', 'Billie Holiday', 'Ella Fitzgerald', 'Lucille Ball', 'Fred Astaire', 'Ginger Rogers', 'Walt Disney', 'Henry Ford', 'Pele', 'Pocahontas', 'Sacagawea', 'Attila the Hun', 'Nicolaus Copernicus', 'Archimedes', 'Roald Dahl', 'C.S. Lewis', 'George R.R. Martin', 'J.D. Salinger', 'Gabriel García Márquez', 'Langston Hughes', 'Zora Neale Hurston', 'Johannes Gutenberg', 'Mary Shelley', 'Victor Hugo', 'Hans Christian Andersen', 'James Joyce', 'Lewis Carroll', 'Jack Kerouac', 'Fyodor Dostoevsky', 'Igor Stravinsky', 'Pyotr Ilyich Tchaikovsky', 'Sergei Rachmaninoff', 'Bram Stoker', 'Leo Tolstoy', 'Marcel Proust', 'Ernest Shackleton', 'Roald Amundsen', 'Neil Gaiman', 'Harper Lee', 'Anne Rice', 'Jorge Luis Borges', 'Joseph Stalin', 'Benito Mussolini', 'Giacomo Casanova', 'Pierre-Auguste Renoir', 'Claude Monet', 'Edgar Degas', 'Auguste Rodin', 'Fyodor Dostoevsky', 'Nikolai Gogol', 'Mikhail Gorbachev', 'Joseph Conrad', 'Franz Schubert', 'Al Capone', 'Jesse James', 'Cleopatra', 'Frederick Douglass', 'H.G. Wells', 'John Keats', 'Henry David Thoreau', 'Ralph Waldo Emerson', 'W.B. Yeats', 'T.S. Eliot', 'Frida Kahlo', 'Vincent van Gogh', 'Franz Kafka', 'Edvard Grieg', 'Richard Wagner', 'Arthur Conan Doyle', 'Alexander Graham Bell', 'Guglielmo Marconi', 'John Milton', 'Geoffrey Chaucer', 'Virginia Woolf', 'Dylan Thomas', 'Sir Walter Raleigh', 'Edmund Spenser', 'Oliver Cromwell', 'Isaac Bashevis Singer', 'Dr. Seuss', 'William Butler Yeats', 'William Wordsworth', 'Samuel Taylor Coleridge', 'Lord Byron', 'Percy Bysshe Shelley', 'Elizabeth Barrett Browning', 'Robert Browning', 'Emily Bronte', 'Charlotte Bronte', 'Anne Bronte', 'Mary Wollstonecraft', 'Henry VIII', 'Catherine the Great', 'Alexander Pushkin', 'Michelangelo', 'Raphael', 'Giotto', 'Jan van Eyck', 'Hieronymus Bosch', 'Sandro Botticelli', 'Titian', 'Caravaggio', 'Johannes Vermeer', 'Rembrandt', 'Francisco Goya', 'Jean-Jacques Rousseau', 'Voltaire', 'Thomas Hobbes', 'John Locke', 'Adam Smith', 'David Hume', 'Auguste Comte', 'Thomas Malthus', 'John Stuart Mill', 'Emile Durkheim', 'Karl Marx', 'Sigmund Freud', 'Max Weber', 'Charlotte Perkins Gilman', 'Thorstein Veblen', 'John Dewey', 'Franz Boas', 'Margaret Mead', 'Ruth Benedict', 'Marshall McLuhan', 'W.E.B. Du Bois', 'Booker T. Washington', 'Harriet Beecher Stowe', 'Upton Sinclair', 'William Lloyd Garrison', 'Nathaniel Hawthorne', 'Donald Trump', 'Elon Musk', 'Hamlet']
PERSONS = list(set(PERSONS))
PLACES = ['New York City', 'Paris', 'Tokyo', 'London', 'Sydney', 'San Francisco', 'Rome', 'Mars', 'Mount Everest', 'The Great Barrier Reef', 'Antarctica', 'The Amazon Rainforest', 'The Sahara Desert', 'Venice', 'Atlantis', 'The Moon', 'Narnia', 'Gotham City', 'Hogwarts', 'Middle Earth', 'Los Angeles', 'Berlin', 'Barcelona', 'Rio de Janeiro', 'Dubai', 'Singapore', 'Hong Kong', 'Chicago', 'Toronto', 'Amsterdam', 'Bangkok', 'Istanbul', 'Cairo', 'Lisbon', 'Vienna', 'Budapest', 'Prague', 'Copenhagen', 'Stockholm', 'Helsinki', 'Oslo', 'Reykjavik', 'Moscow', 'St. Petersburg', 'Athens', 'Warsaw', 'Bucharest', 'Dublin', 'Edinburgh', 'Brussels', 'Geneva', 'Zurich', 'Milan', 'Florence', 'Naples', 'Marseille', 'Lyon', 'Bordeaux', 'Madrid', 'Seville', 'Granada', 'Lima', 'Bogota', 'Caracas', 'Quito', 'Buenos Aires', 'Santiago', 'Melbourne', 'Perth', 'Auckland', 'Wellington', 'Cape Town', 'Johannesburg', 'Nairobi', 'Cairo', 'Casablanca', 'Tunis', 'Dakar', 'Accra', 'Lagos', 'Kigali', 'Addis Ababa', 'Algiers', 'Fez', 'Marrakesh', 'Tangier', 'Tripoli', 'Luxor', 'Petra', 'Jerusalem', 'Tel Aviv', 'Beirut', 'Damascus', 'Amman', 'Ankara', 'Tehran', 'Baghdad', 'Kuwait City', 'Muscat', 'Abu Dhabi', 'Doha', 'Manama', 'Riyadh', 'Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad', 'Pune', 'Jaipur', 'Udaipur', 'Varanasi', 'Agra', 'Kathmandu', 'Lhasa', 'Thimphu', 'Dhaka', 'Colombo', 'Karachi', 'Lahore', 'Islamabad', 'Bishkek', 'Tashkent', 'Ashgabat', 'Dushanbe', 'Kabul', 'Hanoi', 'Ho Chi Minh City', 'Phnom Penh', 'Vientiane', 'Yangon', 'Jakarta', 'Bandung', 'Surabaya', 'Bali', 'Manila', 'Cebu', 'Davao', 'Kuala Lumpur', 'Penang', 'Singapore', 'Brunei', 'Bangkok', 'Chiang Mai', 'Phuket', 'Pattaya', 'Beijing', 'Shanghai', 'Hong Kong', 'Guangzhou', 'Shenzhen', 'Chengdu', 'Hangzhou', 'Wuhan', 'Xi\'an', 'Chongqing', 'Taipei', 'Kaohsiung', 'Tainan', 'Taichung', 'Hualien', 'Seoul', 'Busan', 'Incheon', 'Gwangju', 'Jeju Island', 'Pyongyang', 'Tokyo', 'Osaka', 'Kyoto', 'Hiroshima', 'Nagasaki', 'Sapporo', 'Fukuoka', 'Kobe', 'Yokohama', 'Nagoya', 'Okinawa', 'Ulaanbaatar', 'Vladivostok', 'Siberia', 'Kamchatka', 'Yakutsk', 'Omsk', 'Novosibirsk', 'Kazan', 'Samara', 'Sochi', 'Krasnoyarsk', 'Irkutsk', 'Chelyabinsk', 'Anchorage', 'Fairbanks', 'Juneau', 'Honolulu', 'Hilo', 'Kauai', 'Maui', 'Lahaina', 'Vancouver', 'Calgary', 'Edmonton', 'Winnipeg', 'Toronto', 'Ottawa', 'Montreal', 'Quebec City', 'Halifax', 'St. John\'s', 'Victoria', 'Whistler', 'Tofino', 'Banff', 'Lake Louise', 'Niagara Falls', 'Churchill', 'Yellowknife', 'Iqaluit', 'Whitehorse', 'Nuuk', 'Reykjavik', 'Akureyri', 'Húsavík', 'Vestmannaeyjar', 'Isafjordur', 'Svalbard', 'Longyearbyen', 'Barentsburg', 'Tromsø', 'Trondheim', 'Bergen', 'Stavanger', 'Alesund', 'Geiranger', 'Lofoten Islands', 'Bodø', 'Hammerfest', 'Kirkenes', 'Narvik', 'Stockholm', 'Gothenburg', 'Malmö', 'Uppsala', 'Visby', 'Luleå', 'Kiruna', 'Åre', 'Falun', 'Västerås', 'Jönköping', 'Helsinki', 'Tampere', 'Turku', 'Oulu', 'Rovaniemi', 'Kuopio', 'Jyväskylä', 'Vaasa', 'Joensuu', 'Kotka', 'Mikkeli', 'Lahti', 'Pori', 'Tallinn', 'Tartu', 'Pärnu', 'Narva', 'Viljandi', 'Rakvere', 'Kuressaare', 'Võru', 'Paide', 'Haapsalu', 'Riga', 'Daugavpils', 'Liepaja', 'Jelgava', 'Jurmala', 'Ventspils', 'Rezekne', 'Valmiera', 'Jekabpils', 'Vilnius', 'Kaunas', 'Klaipeda', 'Siauliai', 'Panevezys', 'Alytus', 'Marijampole', 'Mazeikiai', 'Telsiai', 'Utena', 'Kedainiai', 'Warsaw', 'Krakow', 'Wrocław', 'Gdańsk', 'Poznań', 'Łódź', 'Katowice', 'Lublin', 'Toruń', 'Bydgoszcz', 'Białystok', 'Częstochowa', 'Kielce', 'Gdynia', 'Rzeszów', 'Szczecin', 'Olsztyn', 'Zakopane', 'Sopot', 'Berlin', 'Munich', 'Frankfurt', 'Hamburg', 'Cologne', 'Düsseldorf', 'Stuttgart', 'Leipzig', 'Dresden']
PLACES = list(set(PLACES))
# GENRE = ['Essay', 'Poem', 'Rap battle', 'Songs','Horror Story', 'Comedy', 'Romance', 'Science Fiction', 'Thriller', 'Mystery', 'Fantasy', 'Adventure', 'Short Story'] + ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western', 'Epic', 'Gothic', 'Satire', 'Tragedy', 'Tragicomedy', 'Romantic Comedy', 'Dystopian', 'Utopian', 'Cyberpunk', 'Steampunk', 'Dieselpunk', 'Solarpunk', 'Sword and Sorcery', 'High Fantasy', 'Low Fantasy', 'Urban Fantasy', 'Dark Fantasy', 'Hard Science Fiction', 'Soft Science Fiction', 'Space Opera', 'Time Travel', 'Alternate History', 'Military Science Fiction', 'Parallel Universe', 'Post-Apocalyptic', 'Alien Invasion', 'First Contact', 'Artificial Intelligence', 'Superhero', 'Vampire', 'Werewolf', 'Zombie', 'Ghost', 'Occult', 'Detective', 'Espionage', 'Heist', 'Legal', 'Martial Arts', 'Medical', 'Police Procedural', 'Political', 'Psychological', 'Serial Killer', 'Spy', 'Supernatural', 'Survival', 'Techno-thriller', 'True Crime', 'Courtroom', 'Young Adult', 'New Adult', 'Middle Grade', 'Children\'s', 'Picture Book', 'Chick Lit', 'Lad Lit', 'Contemporary', 'Historical Fiction', 'Regency', 'Victorian', 'Medieval', 'Renaissance', 'Ancient World', 'Prehistoric', 'Native American', 'Colonial', 'Civil War', 'World War I', 'World War II', 'Cold War', 'Vietnam War', 'Gulf War', 'Iraq War', 'Afghanistan War', 'Literary Fiction', 'Magical Realism', 'Mythology', 'Fable', 'Fairy Tale', 'Legend', 'Folklore', 'Paranormal', 'Philosophical', 'Religious', 'Inspirational', 'Spiritual', 'Autobiography', 'Memoir', 'Biography', 'Travelogue', 'Diary', 'Journal', 'Anthology', 'Essay', 'Short Story', 'Novella', 'Novel', 'Flash Fiction', 'Micro Fiction', 'Poetry', 'Prose', 'Haiku', 'Sonnet', 'Epic Poem', 'Ballad', 'Limerick', 'Villanelle', 'Sestina', 'Pantoum', 'Ghazal', 'Free Verse', 'Blank Verse', 'Ode', 'Elegy', 'Pastoral', 'Experimental', 'Concrete', 'Graphic Novel', 'Comic Book', 'Manga', 'Manhwa', 'Manhua', 'Webtoon', 'Fanfiction', 'Crossover', 'Alternate Universe', 'Canon-Divergent', 'Missing Scene', 'Fluff', 'Angst', 'Hurt/Comfort', 'Smut', 'PWP', 'Crack', 'Drabble', 'Femslash', 'Slash', 'Gen', 'Het', 'Academic', 'Architecture', 'Art', 'Business', 'Economics', 'Education', 'Engineering', 'Environment', 'Fashion', 'Film', 'Food', 'Gardening', 'Gender Studies', 'Geography', 'Health', 'History', 'Home', 'Humor', 'Journalism', 'Language', 'Law', 'LGBTQ+', 'Linguistics', 'Literary Criticism', 'Mathematics', 'Medicine', 'Military', 'Music', 'Nature', 'Performing Arts', 'Pets', 'Philosophy', 'Photography', 'Physics', 'Political Science', 'Psychology', 'Reference', 'Religion', 'Science', 'Self-Help', 'Social Science', 'Sociology', 'Sports', 'Technology', 'Transportation', 'Travel', 'True Crime', 'Women\'s Studies', 'World Affairs', 'Animal Fiction', 'Bildungsroman', 'Campus Novel', 'Cli-fi', 'Farce', 'Metafiction', 'Picaresque', 'Pastiche', 'Slapstick', 'Space Western', 'Swashbuckler', 'Treasure Hunt', 'Whodunit', 'Hardboiled', 'Noir', 'Cozy Mystery', 'Epistolary', 'Gothic Romance', 'Paranormal Romance', 'Romantic Suspense', 'Erotica', 'BDSM', 'Menage', 'Reverse Harem', 'Taboo', 'Queer', 'Gay', 'Lesbian', 'Bisexual', 'Transgender', 'Non-binary', 'Polyamorous', 'Kink', 'Fetish', 'Role Reversal', 'Disability', 'Mental Health', 'Body Positivity', 'Diversity', 'Own Voices', 'Multicultural', 'Interracial', 'Indigenous', 'Immigrant', 'Refugee', 'Diaspora', 'Coming of Age', 'Loss', 'Grief', 'Revenge', 'Redemption', 'Empowerment', 'Resilience', 'Family Saga', 'Generational', 'Friendship', 'Workplace', 'Retelling', 'Adaptation', 'Pastiche', 'Reimagining', 'Mashup', 'Cultural', 'Regional', 'Rural', 'Urban', 'Small Town', 'Beach Read', 'Chiller', 'Dark Comedy', 'Docudrama', 'Screwball Comedy', 'Melodrama', 'Neorealism', 'Romantic Drama', 'Social Realism', 'Surrealism', 'Tragicomedy', 'Absurdist', 'Black Comedy', 'Horror Comedy', 'Mumblecore', 'Parody', 'Spoof', 'Mockumentary', 'Satirical', 'Silent Film', 'Slasher', 'Splatter', 'Supernatural Horror', 'Survival Horror', 'Zombie Apocalypse', 'Found Footage', 'Psychological Horror', 'Giallo', 'Body Horror', 'B-movie', 'Cult Film', 'Exploitation', 'Grindhouse', 'Hammer Horror', 'Kaiju', 'Nunsploitation', 'Ozploitation', 'Spaghetti Western', 'Tokusatsu', 'Video Nasty', 'Wuxia', 'Gangster', 'Neo-Noir', 'Crime Thriller', 'Prison Film', 'Heist Film', 'Caper', 'Revenge Thriller', 'Spy Thriller', 'Supernatural Thriller', 'Political Thriller', 'Legal Thriller', 'Medical Thriller', 'Action Thriller', 'Erotic Thriller', 'Domestic Thriller', 'Gothic Thriller', 'Techno-Thriller', 'Nature', 'Outdoors', 'Wildlife', 'Ecology', 'Conservation', 'Sustainable Living', 'Gardening', 'Permaculture', 'Urban Gardening', 'Landscape Design', 'Horticulture', 'Botany', 'Adventure Travel']
# GENRE = list(set(GENRE))
AUTHOR = ['J.K. Rowling', 'Ernest Hemingway', 'Arthur Conan Doyle', 'Shakespeare', 'Toni Morrison', 'Margaret Atwood', 'Gabriel Garcia Marquez', 'H.G. Wells', 'Jane Austen', 'Agatha Christie', 'Stephen King', 'Neil Gaiman', 'Mary Shelley', 'Edgar Allan Poe', 'J.R.R. Tolkien', 'Maya Angelou'] + ['William Shakespeare', 'Jane Austen', 'George Orwell', 'Charles Dickens', 'J.K. Rowling', 'Ernest Hemingway', 'F. Scott Fitzgerald', 'Virginia Woolf', 'Mark Twain', 'Leo Tolstoy', 'Fyodor Dostoevsky', 'James Joyce', 'Franz Kafka', 'Oscar Wilde', 'J.R.R. Tolkien', 'C.S. Lewis', 'H.G. Wells', 'Aldous Huxley', 'Herman Melville', 'Joseph Conrad', 'Emily Brontë', 'Charlotte Brontë', 'Anne Brontë', 'Nathaniel Hawthorne', 'Harper Lee', 'John Steinbeck', 'Arthur Conan Doyle', 'Edgar Allan Poe', 'Bram Stoker', 'Mary Shelley', 'H.P. Lovecraft', 'Agatha Christie', 'Ray Bradbury', 'Isaac Asimov', 'Arthur C. Clarke', 'Philip K. Dick', 'George R.R. Martin', 'Stephen King', 'Toni Morrison', 'Alice Walker', 'James Baldwin', 'Zora Neale Hurston', 'Maya Angelou', 'Langston Hughes', 'Ralph Ellison', 'Chinua Achebe', 'Wole Soyinka', 'Gabriel García Márquez', 'Carlos Fuentes', 'Isabel Allende', 'Mario Vargas Llosa', 'Jorge Luis Borges', 'Julio Cortázar', 'Salman Rushdie', 'Arundhati Roy', 'Amitav Ghosh', 'Kazuo Ishiguro', 'Haruki Murakami', 'Yukio Mishima', 'Banana Yoshimoto', 'Kenzaburo Oe', 'Orhan Pamuk', 'Elif Shafak', 'Naguib Mahfouz', 'Doris Lessing', 'Nadine Gordimer', 'J.M. Coetzee', 'Chimamanda Ngozi Adichie', 'Tsitsi Dangarembga', 'Yaa Gyasi', 'Chinelo Okparanta', 'Marlon James', 'Edwidge Danticat', 'Roxane Gay', 'Jamaica Kincaid', 'Junot Díaz', 'Sandra Cisneros', 'Julia Alvarez', 'Margaret Atwood', 'Michael Ondaatje', 'Yann Martel', 'Douglas Adams', 'Neil Gaiman', 'Terry Pratchett', 'Ursula K. Le Guin', 'Kurt Vonnegut', 'Roald Dahl', 'Dr. Seuss', 'Maurice Sendak', 'Madeleine L\'Engle', 'C.S. Lewis', 'Philip Pullman', 'L.M. Montgomery', 'Louisa May Alcott', 'A.A. Milne', 'E.B. White', 'Beatrix Potter', 'Lewis Carroll', 'J.M. Barrie', 'Astrid Lindgren', 'P.L. Travers', 'Shel Silverstein', 'Antoine de Saint-Exupéry', 'E.L. James', 'Danielle Steel', 'Nora Roberts', 'Nicholas Sparks', 'Stephenie Meyer', 'Dan Brown', 'John Grisham', 'Michael Crichton', 'Tom Clancy', 'James Patterson', 'Lee Child', 'Gillian Flynn', 'Sue Grafton', 'Stieg Larsson', 'Ian Fleming', 'Robert Ludlum', 'Patricia Highsmith', 'Ruth Rendell', 'P.D. James', 'Daphne du Maurier', 'Graham Greene', 'Donna Tartt', 'Zadie Smith', 'Ian McEwan', 'Martin Amis', 'Salman Rushdie', 'Iris Murdoch', 'Kingsley Amis', 'Anthony Burgess', 'Evelyn Waugh', 'G.K. Chesterton', 'Thomas Hardy', 'D.H. Lawrence', 'Graham Swift', 'Angela Carter', 'W. Somerset Maugham', 'E.M. Forster', 'A.S. Byatt', 'Hilary Mantel', 'Jeanette Winterson', 'Margaret Drabble', 'Dylan Thomas', 'Ted Hughes', 'Philip Larkin', 'Seamus Heaney', 'W.B. Yeats', 'T.S. Eliot', 'Sylvia Plath', 'Anne Sexton', 'Robert Frost', 'Langston Hughes', 'Walt Whitman', 'Emily Dickinson', 'Edgar Allan Poe', 'Allen Ginsberg', 'E.E. Cummings', 'Jack Kerouac', 'William S. Burroughs', 'Charles Bukowski', 'Harper Lee', 'Truman Capote', 'Tennessee Williams', 'Carson McCullers', 'Flannery O\'Connor', 'Eudora Welty', 'William Faulkner', 'Ernest Hemingway', 'F. Scott Fitzgerald', 'Gertrude Stein', 'Djuna Barnes', 'Anais Nin', 'Henry Miller', 'Herman Hesse', 'Thomas Mann', 'Günter Grass', 'Heinrich Böll', 'Erich Maria Remarque', 'Friedrich Dürrenmatt', 'Max Frisch', 'Bertolt Brecht', 'Franz Werfel', 'Stefan Zweig', 'Robert Musil', 'Thomas Bernhard', 'Peter Handke', 'Elfriede Jelinek', 'Milan Kundera', 'Bohumil Hrabal', 'Jaroslav Hašek', 'Karel Čapek', 'Vladimir Nabokov', 'Mikhail Bulgakov', 'Anton Chekhov', 'Leo Tolstoy', 'Fyodor Dostoevsky', 'Ivan Turgenev', 'Nikolai Gogol', 'Aleksandr Solzhenitsyn', 'Boris Pasternak', 'Isaac Babel', 'Vasily Grossman', 'Yevgeny Zamyatin', 'Andrei Platonov', 'Mikhail Lermontov', 'Henrik Ibsen', 'August Strindberg', 'Knut Hamsun', 'Sigrid Undset', 'Selma Lagerlöf', 'Halldór Laxness', 'Gunnar Ekelöf', 'Tomas Tranströmer', 'Edith Södergran', 'Karl Ove Knausgård', 'Jon Fosse', 'Henning Mankell', 'Stieg Larsson', 'Kjell Westö', 'Sofi Oksanen', 'Arto Paasilinna', 'Tove Jansson', 'Mika Waltari', 'Frans Eemil Sillanpää', 'Eino Leino', 'Aksel Sandemose', 'Tarjei Vesaas', 'Karin Fossum', 'Kjell Askildsen', 'Dag Solstad', 'Per Petterson', 'Jo Nesbø', 'Karl Ove Knausgård', 'Espen Stueland', 'Ragnar Hovland', 'Jan Kjærstad']
AUTHOR = list(set(AUTHOR))
NOUN = ['apple', 'chair', 'dog', 'book', 'sun', 'mountain', 'ocean', 'phone', 'cookie', 'guitar', 'bicycle', 'tree', 'camera', 'shoe', 'pencil', 'pillow', 'flower', 'car', 'keyboard', 'house'] +  ['apple', 'ball', 'cat', 'dog', 'elephant', 'flower', 'guitar', 'house', 'island', 'juice', 'kite', 'lion', 'mountain', 'notebook', 'ocean', 'pencil', 'queen', 'rain', 'sun', 'tree', 'umbrella', 'violin', 'whale', 'xylophone', 'yarn', 'zebra', 'airplane', 'boat', 'car', 'drum', 'egg', 'forest', 'giraffe', 'hat', 'ice', 'jacket', 'kangaroo', 'lamp', 'moon', 'nose', 'orange', 'penguin', 'quill', 'river', 'snake', 'train', 'unicorn', 'volcano', 'window', 'x-ray', 'yak', 'zoo', 'actor', 'beach', 'city', 'desert', 'envelope', 'finger', 'glass', 'hair', 'igloo', 'jar', 'key', 'leaf', 'mirror', 'nest', 'octopus', 'piano', 'quilt', 'rose', 'star', 'tiger', 'upstairs', 'vegetable', 'wind', 'xylophonist', 'yard', 'zipper', 'ant', 'bread', 'candle', 'dinosaur', 'eagle', 'fish', 'goat', 'horse', 'iceberg', 'jelly', 'knife', 'ladder', 'mailbox', 'newspaper', 'owl', 'pear', 'question', 'robot', 'spider', 'telephone', 'uncle', 'vase', 'wall', 'xenophobia', 'yogurt', 'zeppelin', 'airport', 'bicycle', 'castle', 'door', 'engine', 'factory', 'garden', 'hotel', 'island', 'jewel', 'kitchen', 'lighthouse', 'museum', 'needle', 'orchard', 'parrot', 'quicksand', 'rocket', 'ship', 'tower', 'universe', 'valley', 'waterfall', 'xylograph', 'year', 'zigzag', 'arm', 'basket', 'camera', 'desk', 'elevator', 'flag', 'glove', 'hamster', 'ink', 'jungle', 'king', 'lock', 'map', 'net', 'ostrich', 'paint', 'queen', 'ring', 'scissors', 'television', 'underground', 'victory', 'wheel', 'xenon', 'yellow', 'zephyr', 'alarm', 'box', 'cactus', 'dolphin', 'eye', 'feather', 'grape', 'hospital', 'iron', 'jigsaw', 'keyboard', 'library', 'monkey', 'night', 'oardvark', 'pillow', 'quiver', 'sand', 'table', 'urchin', 'vulture', 'winter', 'xylography', 'yawn', 'zigzagging', 'air', 'brick', 'cup', 'dream', 'eggplant', 'flute', 'gate', 'hedgehog', 'iguanodon', 'jellyfish', 'koala', 'lake', 'mountain', 'ninja', 'octagon', 'plate', 'quasar', 'rainbow', 'squirrel', 'tornado', 'utensil', 'viaduct', 'wombat', 'xenolith', 'yacht', 'ziggurat', 'accordion', 'banana', 'compass', 'daisy', 'eraser', 'fern', 'gazelle', 'harbor', 'ice cream', 'jogger', 'kettle', 'llama', 'magnet', 'nebula', 'oboe', 'pumpkin', 'quokka', 'raccoon', 'seashell', 'tambourine', 'urchin', 'vanilla', 'walrus', 'xerophyte', 'yoga', 'zucchini', 'aardvark', 'butterfly', 'crescent', 'daffodil', 'escalator', 'fire', 'gopher', 'hippopotamus', 'insect', 'jackal', 'kelp', 'lagoon', 'meadow', 'nectarine', 'otter', 'palm', 'quinoa', 'rhinoceros', 'satellite', 'tulip', 'vacuum', 'watermelon', 'xiphias', 'yurt', 'zenith', 'alchemy', 'balloon', 'carnival', 'dandelion', 'equator', 'fjord', 'galaxy', 'hurricane', 'iguana', 'jester', 'kiosk', 'labyrinth', 'mongoose', 'neutron', 'obelisk', 'plankton', 'quasar', 'reptile', 'solstice', 'tectonic', 'utopia', 'venom', 'wavelength', 'xenophobe', 'yucca', 'zeppelin', 'aquarium', 'bagpipe', 'carousel', 'dome', 'eclipse', 'fossil', 'geode', 'hologram', 'illusion', 'jasmine', 'kaleidoscope', 'lantern', 'metronome', 'nebula', 'origami', 'prism', 'quail', 'rune', 'saxophone', 'typewriter', 'ukulele', 'veranda', 'windmill', 'xenomorph', 'yak', 'zodiac', 'ambrosia', 'bonsai', 'chameleon', 'dove', 'ember', 'falcon', 'geyser', 'hyacinth', 'iris', 'jade', 'kiwi', 'lilac', 'mandolin', 'narwhal', 'oardvark', 'peacock', 'quilt', 'raven', 'sphinx', 'toucan', 'vortex', 'willow', 'xylophonist', 'yew', 'zen', 'alchemy', 'breeze', 'coral', 'dragonfly', 'ember', 'fig', 'goblin', 'hail', 'icicle', 'jackrabbit', 'kelp', 'lemur', 'marigold', 'nymph', 'oasis', 'pyramid', 'quicksilver', 'serpent', 'talisman', 'unicorn', 'vixen', 'wisteria', 'xenops', 'yeti', 'zephyr']
NOUN = list(set(NOUN))
ADJECTIVE = ['abandoned', 'able', 'absolute', 'adorable', 'adventurous', 'academic', 'acceptable', 'acclaimed', 'accomplished', 'accurate', 'aching', 'acidic', 'acrobatic', 'adorable', 'adventurous', 'affectionate', 'afraid', 'aged', 'aggressive', 'agile', 'agreeable', 'ajar', 'alarmed', 'alarming', 'alert', 'alienated', 'alive', 'all', 'alleged', 'amused', 'angry', 'annoyed', 'anxious', 'apprehensive', 'arbitrary', 'arrogant', 'artistic', 'ashamed', 'assertive', 'astonishing', 'athletic', 'attractive', 'average', 'awesome', 'awful', 'awkward', 'azure', 'babyish', 'back', 'bad', 'bare', 'barren', 'bashful', 'beautiful', 'belated', 'beneficial', 'better', 'bewitched', 'big', 'billowing', 'bitter', 'black', 'bland', 'blank', 'blaring', 'bleak', 'blind', 'blissful', 'blond', 'blue', 'blushing', 'bold', 'bored', 'boring', 'bouncy', 'brave', 'brief', 'bright', 'brilliant', 'brisk', 'broken', 'bronze', 'brown', 'bruised', 'bubbly', 'bulky', 'bumpy', 'buoyant', 'burdensome', 'burly', 'bustling', 'busy', 'buttery', 'buzzing', 'calm', 'candid', 'canine', 'capital', 'carefree', 'careful', 'careless', 'caring', 'cautious', 'cavernous', 'celebrated', 'charming', 'cheap', 'cheerful', 'cheery', 'chief', 'chilly', 'chubby', 'circular', 'classic', 'clean', 'clear', 'clever', 'close', 'closed', 'cloudy', 'clueless', 'clumsy', 'cluttered', 'coarse', 'cold', 'colorful', 'colorless', 'colossal', 'comfortable', 'common', 'compassionate', 'competent', 'complete', 'complex', 'complicated', 'composed', 'concerned', 'concrete', 'confused', 'conscious', 'considerate', 'constant', 'content', 'conventional', 'cooked', 'cool', 'cooperative', 'coordinated', 'corny', 'corrupt', 'costly', 'courageous', 'courteous', 'crafty', 'crazy', 'creamy', 'creative', 'creepy', 'criminal', 'crisp', 'critical', 'crooked', 'crowded', 'cruel', 'crushing', 'cuddly', 'cultivated', 'cultured', 'cumbersome', 'curly', 'curvy', 'cute', 'cylindrical', 'damaged', 'damp', 'dangerous', 'dapper', 'daring', 'dark', 'darling', 'dazzling', 'dead', 'deadly', 'deafening', 'dear', 'dearest', 'decent', 'decimal', 'decisive', 'deep', 'defenseless', 'defensive', 'defiant', 'deficient', 'definite', 'definitive', 'delayed', 'delectable', 'delicious', 'delightful', 'delirious', 'demanding', 'dense', 'dental', 'dependable', 'dependent', 'descriptive', 'deserted', 'detailed', 'determined', 'devoted', 'different', 'difficult', 'digital', 'diligent', 'dim', 'dimpled', 'dimwitted', 'direct', 'disastrous', 'discrete', 'disfigured', 'disgusting', 'disloyal', 'dismal', 'distant', 'distant', 'distinct', 'distorted', 'dizzy', 'dopey', 'doting', 'double', 'downright', 'drab', 'draconian', 'dramatic', 'dreary', 'droopy', 'dry', 'dual', 'dull', 'dutiful', 'eager', 'earnest', 'early', 'easy', 'easy-going', 'ecstatic', 'edible', 'educated', 'elaborate', 'elastic', 'elated', 'elderly', 'electric', 'elegant', 'elementary', 'elliptical', 'emaciated', 'embarrassed', 'embellished', 'eminent', 'emotional', 'empty', 'enchanted', 'enchanting', 'energetic', 'enlightened', 'enormous', 'enraged', 'entertaining', 'enthusiastic', 'enviable', 'equal', 'equatorial', 'essential', 'esteemed', 'ethereal', 'ethical', 'euphoric', 'even', 'evergreen', 'everlasting', 'every', 'evil', 'exalted', 'excellent', 'exemplary', 'exhausted', 'excitable', 'excited', 'exciting', 'exotic', 'expensive', 'experienced', 'expert', 'extraneous', 'extroverted', 'fabulous', 'failing', 'faint', 'fair', 'faithful', 'fake', 'false', 'familiar', 'famous', 'fancy', 'fantastic', 'far', 'faraway', 'fast', 'fat', 'fatal', 'fatherly', 'favorable', 'favorite', 'fearful', 'fearless', 'feisty', 'feline', 'female', 'feminine', 'few', 'fickle', 'filthy', 'fine', 'finished', 'firm', 'first', 'firsthand', 'fitting', 'fixed', 'flaky', 'flamboyant', 'flashy', 'flat', 'flawed', 'flawless', 'flickering', 'flimsy', 'flippant', 'flowery', 'fluffy', 'fluid', 'flustered', 'focused', 'fond', 'foolhardy', 'foolish', 'forceful', 'forked', 'formal', 'forsaken', 'forthright', 'fortunate', 'fragrant', 'frail', 'frank', 'frayed', 'free', 'French', 'fresh', 'frequent', 'friendly', 'frightened', 'frightening', 'frigid', 'frilly', 'frivolous', 'frizzy', 'frosty', 'frozen', 'frugal', 'fruitful', 'full', 'fumbling', 'functional', 'funny', 'fussy', 'fuzzy', 'gargantuan', 'gaseous', 'general', 'generous']

sentences = []
documents = []
def random_subsample(array_in):
    max_items = 12
    np.random.seed(0)
    subset_idxs = np.random.choice(len(array_in), size=max_items, replace=False).tolist()
    array_in = [array_in[i] for i in subset_idxs]

    return array_in
for idx, genre in enumerate(['Essay', 'Poem', 'Song']):
    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "

        sentence = f"Write me a {modifier}{combo[2]} about a {combo[5]} {combo[4]} who meets {combo[0]} in {combo[1]} in the style of {combo[3]}"
        sentences.append(sentence)

    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "

        sentence = f"Compose a {modifier}{combo[2]} set in {combo[1]}, where a {combo[5]} {combo[4]} encounters {combo[0]}, inspired by the works of {combo[3]}."
        sentences.append(sentence)

    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Imagine a {modifier}{combo[2]} about {combo[0]} who discover a {combo[5]} {combo[4]} in {combo[1]}"
        sentences.append(sentence)
        
    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Write a {modifier}{combo[2]} that follows the adventures of {combo[0]} in {combo[1]} as they seek a {combo[5]} {combo[4]}"
        sentences.append(sentence)

    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Craft a {modifier}{combo[2]} in which {combo[0]} explore {combo[1]} and come across a {combo[5]} {combo[4]}, with literary elements drawn from {combo[3]}."
        sentences.append(sentence)

    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Weave a {modifier}{combo[2]} where {combo[0]} uncovers a {combo[5]} {combo[4]} in {combo[1]}, emulating the style of {combo[3]}."
        sentences.append(sentence)

    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Create a {modifier}{combo[2]} in which {combo[0]} encounter a {combo[5]} {combo[4]} while traversing {combo[1]}, drawing inspiration from {combo[3]}."
        sentences.append(sentence)

    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Pen a {modifier}{combo[2]} that tells the story of {combo[0]}"
        sentences.append(sentence)

    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Write a {modifier}{combo[2]} featuring {combo[0]}"
        sentences.append(sentence)

    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Imagine a {modifier}{combo[2]}, where {combo[0]} stumble upon a {combo[5]} {combo[4]} in {combo[1]}, with dialogue and atmosphere inspired by {combo[3]}."
        sentences.append(sentence)

    for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PLACES), [genre], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Craft a {modifier}{combo[2]} that delves into the lives of {combo[0]}"
        sentences.append(sentence)

    print(sentences[0])
    unique_sentences = list(set(sentences))
    print(f"There are {len(unique_sentences)} unique combinations.")
    print(unique_sentences[:5])

    np.random.shuffle(unique_sentences)
    np.random.seed(idx)
    max_documents = 100_000
    subset_idxs = np.random.choice(len(unique_sentences), size=max_documents, replace=False).tolist()
    new_docs = [unique_sentences[i] for i in subset_idxs]

    for doc in new_docs:
        documents.append(doc)


#Generating Rap battles seperately due to the different prompt formats
for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PERSONS), ['Rap Battle'], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Craft a {modifier}{combo[2]} between {combo[0]} and {combo[1]}"
        sentences.append(sentence)

unique_sentences = list(set(sentences))
print(f"There are {len(unique_sentences)} unique combinations.")
print(unique_sentences[:5])

np.random.shuffle(unique_sentences)
np.random.seed(idx)
max_documents = 100_000
subset_idxs = np.random.choice(len(unique_sentences), size=max_documents, replace=False).tolist()
new_docs = [unique_sentences[i] for i in subset_idxs]

for doc in new_docs:
    documents.append(doc)

#Generating Emails seperately due to different prompt formats
for combo in tqdm(itertools.product(random_subsample(PERSONS), random_subsample(PERSONS), ['Email'], random_subsample(AUTHOR), random_subsample(NOUN), random_subsample(ADJECTIVE))):
        modifier = ""
        randint = random.randint(0, 2)
        if randint == 0:
            modifier = "long "
        elif random.randint == 1:
            modifier = "short "
            
        sentence = f"Craft a {modifier}{combo[2]} between {combo[0]} and {combo[1]}"
        sentences.append(sentence)

unique_sentences = list(set(sentences))
print(f"There are {len(unique_sentences)} unique combinations.")
print(unique_sentences[:5])

np.random.shuffle(unique_sentences)
np.random.seed(idx)
max_documents = 100_000
subset_idxs = np.random.choice(len(unique_sentences), size=max_documents, replace=False).tolist()
new_docs = [unique_sentences[i] for i in subset_idxs]

for doc in new_docs:
    documents.append(doc)

documents = [{"00": doc, "00_len": len(doc)} for doc in documents]

atlas.map_text(
    data=documents,
    indexed_field='00',
    name='generated creative dataset v9',
    reset_project_if_exists=True,
)
