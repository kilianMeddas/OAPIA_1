# Import nécéssaire
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import tempfile
import os
import fitz 
import re
import csv
import fitz
import pymupdf 
import os  
import io 
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer,  BitsAndBytesConfig, pipeline, LlamaForCausalLM
import torch
import time
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
warnings.filterwarnings("ignore")

# Pour calculer l'inference
t0 = time.time()

# Fonction pour formater
def correct_encoding(text):
            corrections = {
                'Ã©': 'é',
                'Ã¨': 'è',
                'Ã¢': 'â',
                'Ã´': 'ô',
                'Ãª': 'ê',
                'Å“': 'œ',
                'Ã': 'à',
                'Ã§': 'ç',
                'Ã¹': 'ù',
                'Ã»': 'û',
                'Ã‰': 'É',
                'Ã€': 'À',
                'â€™': "'",
                'â€“': '-',
                'â€œ': '“',
                'â€': '”',
                'Ã¶': 'ö',
                'Ã¯': 'ï',
                'Ã¼': 'ü',
                'â€¦': '…',
                'Â': '',
            }
            for wrong, correct in corrections.items():
                text = text.replace(wrong, correct)
            return text

# Fonction pour extraire les images
def recoverpix(doc, item):
    xref = item[0]  # Référence de l'image dans le PDF
    return doc.extract_image(xref)  # Extraire l'imag

# Titre du QCM
st.header("QCM : L'IA au service de la sécurité intérieure", divider=True)

# Déclaration variables streamlit 
if 'commencer' not in st.session_state:
    st.session_state['commencer'] = False
if 'next' not in st.session_state:
    st.session_state['next'] = False 

if 'end' not in st.session_state:
    st.session_state['end'] = False 

if 'quest_num' not in st.session_state:
    st.session_state['quest_num'] = 0

st.session_state.setdefault('point', 0)
st.session_state.setdefault('nom', None)
st.session_state.setdefault('prenom', None)
st.session_state.setdefault('end', False)
st.session_state.setdefault('dropped', False)
st.session_state.setdefault('uploaded_files', None)
st.session_state.setdefault('gene', None)

# Variables Phi3 vision
st.session_state.setdefault('para_phi3', False)
st.session_state.setdefault('processor', None)
st.session_state.setdefault('prompt_phi', None)
st.session_state.setdefault('phi_model', None)
st.session_state.setdefault('messages', None)
st.session_state.setdefault('para_llama3', None)

# Variable Llama3
st.session_state.setdefault('liste', None)
st.session_state.setdefault('tokenizer', None)
st.session_state.setdefault('chunked_documents', None)
st.session_state.setdefault('llama_model', None)
st.session_state.setdefault('para_llama3', None)
st.session_state.setdefault('reponse', None)
st.session_state.setdefault('inputs', None)

# Définition des limites 
# limite minimale des images à extraire (en pixels)
dimlimit_width = 0
dimlimit_height = 0

# Ratio minimum de la taille de l'image par rapport à la taille de la page
size_min = 0

# Taille absolue minimale des images à extraire (en octets)
abssize = 0

# Pemret de séparer un texte (format str) en différents chunk selonde des seprators (de gauche à droite)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])

# Quantization en 8bits
nf8_config = BitsAndBytesConfig(load_in_8bit=True)

# Path des modèles utilisées
phi_path = "/workspace/Phi-3-vision-128k-instruct"
llama_path = "/workspace/Meta-Llama-3-8B-Instruct"


# Condition pour charger Phi3 Visions. Cela permet de ne charger ce modèle qu'une fois
if not st.session_state.para_phi3:
    ###########################################################
    ''' Parametrage du model phi 3 Vision'''
    
    st.session_state.processor = AutoProcessor.from_pretrained(phi_path, trust_remote_code=True)

    # Modèle avec quantisation 
    st.session_state.phi_model = AutoModelForCausalLM.from_pretrained(phi_path, attn_implementation="flash_attention_2", 
                                                device_map="cuda", trust_remote_code=True, torch_dtype="auto", 
                                                quantization_config=nf8_config)
    
    # Prompt utilisé pour la description d'image
    st.session_state.messages = [{"role": "user", "content": "\Décris l'image. Tu ne dois rien inerprêter\n<|image_1|>"},
                {"role": "system", 'content': 'français'}]

    # Tokenization
    st.session_state.prompt_phi = st.session_state.processor.tokenizer.apply_chat_template(st.session_state.messages, tokenize=False, add_generation_prompt=True)
    
    # Pour ne plus entrer dans la condition
    st.session_state.para_phi3 = True


# Permettre de supprimr le drag and drop : widget permettant de mêttre les fichiers voulues pour générer des questions dessus
if not st.session_state.dropped:
    st.session_state.uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True, type=['pdf','txt'])

# Liste qui récupère tous les fichiers envoyés
files = []

# Entrer dans la boucle streamlit
if st.session_state.uploaded_files:
    st.session_state.dropped = True
    # Créez un répertoire temporaire
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in st.session_state.uploaded_files:
            # Obtenez le contenu du fichier en mémoire
            bytes_data = uploaded_file.getvalue()
            
            # Définissez le chemin de sauvegarde local dans le répertoire temporaire
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Écrivez le fichier sur le disque
            with open(file_path, 'wb') as f:
                f.write(bytes_data)
            
            # Affichez le nom du fichier et utilisez le viewer PDF
            st.write("Filename:", uploaded_file.name)
            # pdf_viewer(input=bytes_data, width=700)
            
            # Affichez le chemin d'accès du fichier
            st.write("File path:", file_path)
            files.append(file_path)
            if uploaded_file == st.session_state.uploaded_files[-1]:
                st.session_state.gene = True
        if st.session_state.gene:
            for file in files: 
                doc = fitz.open(file)
                page_count = doc.page_count
                xreflist = []  # Liste des refs déjà traités
                imglist = []  # Liste des références des images trouvées pour éviter de reprendre les mêmes

                # Variable pour envoyer les textes à Llama3
                all_pages = []

                # Boucle pour traiter chaque page de chaque fichier
                for pno in range(page_count):
                    st.write("Page", pno+1, "sur", page_count)
                    il = doc.get_page_images(pno)  # Obtenir la liste des images sur la page
                    imglist.extend([x[0] for x in il])  # Ajouter les références des images à la liste

                    number_img = 0 
                    text =''
                    width = height = n = None
                    xref = None
                    imgdata = img_io = image = None
                    inputs = generate_ids = None
                    response = ''
                
                    page = doc.load_page(pno)
                    page_width = page.rect.width
                    page_height = page.rect.height
                    size_max = page_width*page_height*0.35

                    for img in il:
                        number_img += 1
                        xref = img[0]
                        if xref in xreflist:  # Ignorer l'image si elle a déjà été extraite
                            continue

                        width = img[2]
                        height = img[3] 

                        # if width <= dimlimit_width or height <= dimlimit_height:  # Ignorer les petites images
                        #     continue

                        image = recoverpix(doc, img)  # Récupérer l'image avec la fonction définie plus haut 

                        imgdata = image["image"]

                        if width * height <= size_min:  # Ignorer les images avec un faible ratio de taille
                            continue

                        img_io = io.BytesIO(imgdata)
                        xreflist.append(xref)  # Ajouter la référence de l'image à la liste des images extraites

                        image = Image.open(img_io)
                    
                        # Process prompt and image for model input
                        inputs = st.session_state.processor(st.session_state.prompt_phi, [image], return_tensors="pt").to('cuda')

                        # Generate text response using model
                        generate_ids = st.session_state.phi_model.generate(**inputs, eos_token_id=st.session_state.processor.tokenizer.eos_token_id, max_new_tokens=500, do_sample=False)
                        # Remove input tokens from generated response
                        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

                        # Decode generated IDs to text
                        response = st.session_state.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        
                        filename = f"/workspace/Assets/Cours_NTECH/Semaine 1 CNFPJ/J2/test1/"
                        imgfile = os.path.join(filename, f"page{pno+1}_tag{xref}_img{number_img}.png")  # Pour différencier les images
                        fout = open(imgfile, "wb")
                        fout.write(imgdata)  # Écrire les données de l'image dans le fichier
                        fout.close()
                        if response != '':
                            text += "Texte remplaçant une image: " + response + '\n\n'
                        torch.cuda.empty_cache()


                    page = doc.load_page(pno)
                    text += page.get_text()  
                    all_pages.append(text)

                    filename = f"/workspace/Assets/Cours_NTECH/Semaine 1 CNFPJ/J2/test_streamlit/page{pno+1}.txt"
                    
                    with open(filename, "w") as file:
                        file.write(text)
                print(all_pages)
                document = '\n'.join(all_pages)

                imglist = list(set(imglist))  # Retirer les doublons des images trouvées

                st.write(len(set(imglist)), "images in total")  # Afficher le nombre total d'images trouvées
                st.write(len(xreflist), "images extracted")  # Afficher le nombre total d'images extraites

                st.write("#### Tous les documents sont extraits ####")
                mid_time = time.time()
                st.write("Inference time: {}".format(mid_time - t0))
                torch.cuda.empty_cache()

                if not st.session_state.para_llama3:
                    '''Parametrage LLama3'''
                    
                    st.session_state.llama_model = LlamaForCausalLM.from_pretrained(llama_path, attn_implementation="flash_attention_2",
                                                                device_map="cuda", torch_dtype="auto", 
                                                                quantization_config=nf8_config)

                    # Tokenizer #
                    st.session_state.tokenizer = AutoTokenizer.from_pretrained(llama_path)
                    st.session_state.tokenizer.pad_token = st.session_state.tokenizer.eos_token
                    st.session_state.tokenizer.padding_side = "right"
                    st.session_state.llama_model.generation_config.pad_token_id = st.session_state.tokenizer.pad_token_id

                    st.session_state.para_llama3 = True 

                chunked_documents = text_splitter.split_text(open("/workspace/Assets/Cours_NTECH/Semaine 1 CNFPJ/J2/test1/document_final.txt", "r").read())
                length = len(chunked_documents)
                liste = []
                for i in range(length):
                    
                    f_prompt = f"""A partir de ce texte: {chunked_documents[i]}, 
                        Crée un QCM en français, basé uniquement sur le contexte donné, n'invente auqu'une questions ni réponse qui ne seraient pas dans ce contexte:  .
                        Crée une liste de question avec leurs numéros, Chaque question doit avoir 4 possibilités. Question à choix unique aveec réponse donnée à la fin
                    Suis le format suivant: 
                    ##Question: [La question]

                    ##Options:
                    A. [Option A]
                    B. [Option B]
                    C. [Option C]
                    D. [Option D]
                    
                    ##Reponse: [lettre de la bonne réponse]"""
                    inputs = st.session_state.tokenizer(f_prompt, return_tensors="pt").to("cuda")
                    reponse = st.session_state.llama_model.generate(inputs.input_ids, max_new_tokens=1000)
                    reponse = st.session_state.tokenizer.batch_decode(reponse, skip_special_tokens=True)[0]
                    reponse = reponse[len(f_prompt):].strip()
                    liste.append(reponse)
                    torch.cuda.empty_cache()

                    st.write(i+1, 'generation sur ', length)
                st.session_state.gene = True

            liste = '\n'.join(liste)

            question = "Quel..*?\?\n"
            option = r'Options:\n\s*A\.\s*.+\n\s*B\.\s*.+\n\s*C\.\s*.+\n\s*D\.\s*.+'
            reponse =  r'Reponse..\s*\w+'

            liste_question = re.findall(question, liste)
            liste_option = re.findall(option, liste)
            liste_reponse = re.findall(reponse, liste)

            liste_finale = []

            for i in range(len(liste_question)):

                liste_finale.append(liste_question[i])

                try: 
                    liste_finale.append(liste_option[i])
                except: 
                    liste_finale.pop()
                    break

                try:
                    liste_finale.append(liste_reponse[i]+'\n')
                except: 
                    liste_finale.pop()
                    liste_finale.pop()
                    break

            liste_finale = '\n'.join(liste_finale)


            x = correct_encoding(liste_finale)

            test = r"Quel..*?\?\n\s*Options:\n\s*A\.\s*\w+\n\s*B\.\s*\w+\n\s*C\.\s*\w+\n\s*D\.\s*\w+\n\s*Reponse..\s*\w+"
            liste_question = re.findall(test, x)
            dico = {}
            for indexe, question in enumerate(liste_question):
                dico[f'question{indexe}']= re.search("Quel..*?\?\n", question)[0]
                dico[f'option{indexe}'] = re.search(r'Options:\n\s*A\.\s*\w+\n\s*B\.\s*\w+\n\s*C\.\s*\w+\n\s*D\.\s*\w+',question)[0].split('\n')
                dico[f'option{indexe}'] = [value.strip() for value in dico[f'option{indexe}'][1:]]
                dico[f'reponse{indexe}'] = re.search(r'Reponse..\s*\w+', question)[0]

            


            if not st.session_state.commencer:
                st.session_state.nom=st.text_input("Votre nom :")
                st.session_state.prenom=st.text_input("Votre prénom :")
                button_start = st.button(label = "Commencer", key='commence')

                if button_start:
                    st.session_state.commencer = True

            if st.session_state.commencer:

                if st.session_state.next:
                            st.session_state.quest_num += 1
                            st.session_state.next = False
                            st.session_state.bouton_clique = True
                    
                if st.session_state.quest_num <= 10:
                        
                        st.session_state.repondu = False
                        envoyer = st.button('Envoyer')
                        if envoyer and not st.session_state.repondu:
                            st.session_state.repondu = True
                        quest_num = st.session_state.quest_num+1
                        question = st.write('**Question ' +str(quest_num) + ' :**\n\n ' , dico[f'question{st.session_state.quest_num}'])
                        st.write('Choix: ')

                        option1 = st.checkbox(label=dico[f'option{st.session_state.quest_num}'][0], key=f'A{st.session_state.quest_num}', disabled=st.session_state.repondu)
                        option2 = st.checkbox(label=dico[f'option{st.session_state.quest_num}'][1], key=f'B{st.session_state.quest_num}', disabled=st.session_state.repondu)
                        option3 = st.checkbox(label=dico[f'option{st.session_state.quest_num}'][2], key=f'C{st.session_state.quest_num}', disabled=st.session_state.repondu)
                        option4 = st.checkbox(label=dico[f'option{st.session_state.quest_num}'][3], key=f'D{st.session_state.quest_num}', disabled=st.session_state.repondu)
                        
                        if st.button('Question suivante') :
                            st.session_state.next = True

                        

                        # Si le bouton "Envoyer" est cliqué et que la réponse n'a pas encore été envoyée
                        

                        
                        # To give a point if good answer
                        if (option1 and envoyer) and (dico[f'option{st.session_state.quest_num}'][0][:1].strip() in  dico[f'reponse{st.session_state.quest_num}']):
                            st.session_state.point += 1

                        elif (option2 and envoyer) and (dico[f'option{st.session_state.quest_num}'][1][:1].strip() in  dico[f'reponse{st.session_state.quest_num}']):
                            st.session_state.point += 1

                        elif (option3 and envoyer) and (dico[f'option{st.session_state.quest_num}'][2][:1].strip() in  dico[f'reponse{st.session_state.quest_num}']):
                            st.session_state.point += 1

                        elif (option4 and envoyer) and (dico[f'option{st.session_state.quest_num}'][3][:1].strip() in  dico[f'reponse{st.session_state.quest_num}']):
                            st.session_state.point += 1

                        # Si la réponse a déjà été envoyée, afficher la réponse correcte
                        if st.session_state.repondu:
                            rep = dico[f'reponse{st.session_state.quest_num}']
                            st.write(f'**Réponse** : {rep}')

                        st.markdown(f"<h1 style='font-size: 30px;'> Note : {st.session_state.point}/{len(dico)}</h1>", unsafe_allow_html=True)
                

                        

                else:
                    st.write(f'Le test est fini. Note finale : {st.session_state.point}/{len(dico)}')
                    st.session_state.end = True

            if st.session_state.end:
                # Charger ou créer le fichier CSV pour enregistrer les résultats
                csv_file = r'C:\Users\eloua\OneDrive\Bureau\IA_pedago\Projet_final\formatag\resultats.csv'
                
            with open(csv_file, 'w', newline='') as csvfile:
                line = csv.writer(csvfile)
                line.writerow(['Nom', 'Prénom', 'Résultat'])
                line.writerow([st.session_state.nom, st.session_state.prenom,st.session_state.point])


