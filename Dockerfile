FROM python:3.9
COPY . .
RUN sudo . env.sh
RUN pip -r install requiremnents.txt
CMD python algorithm.py