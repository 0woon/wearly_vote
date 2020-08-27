from django import forms
from .models import User



class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ("name","age","gender",
                  "imageScore1","imageScore2", "imageScore3","imageScore4","imageScore5","imageScore6","imageScore7","imageScore8","imageScore9","imageScore10",
                  "imageScore11", "imageScore12", "imageScore13", "imageScore14", "imageScore15", "imageScore16",
                  "imageScore17", "imageScore18", "imageScore19", "imageScore20",
                  "imageScore21", "imageScore22", "imageScore23", "imageScore24", "imageScore25", "imageScore26",
                  "imageScore27", "imageScore28", "imageScore29", "imageScore30",
                  "imageScore31", "imageScore32", "imageScore33", "imageScore34", "imageScore35", "imageScore36",
                  "imageScore37", "imageScore38", "imageScore39", "imageScore40",
                  "imageScore41", "imageScore42", "imageScore43", "imageScore44", "imageScore45", "imageScore46",
                  "imageScore47", "imageScore48", "imageScore49", "imageScore50",
                  "imageScore51", "imageScore52", "imageScore53", "imageScore54", "imageScore55", "imageScore56",
                  "imageScore57", "imageScore58", "imageScore59", "imageScore60",
                  "imageScore61", "imageScore62", "imageScore63", "imageScore64", "imageScore65", "imageScore66",
                  "imageScore67", "imageScore68", "imageScore69", "imageScore70",
                  "imageScore71", "imageScore72", "imageScore73", "imageScore74", "imageScore75", "imageScore76",
                  "imageScore77", "imageScore78", "imageScore79", "imageScore80",
                  "imageScore81", "imageScore82", "imageScore83", "imageScore84", "imageScore85", "imageScore86",
                  "imageScore87", "imageScore88", "imageScore89", "imageScore90",
                  "imageScore91", "imageScore92", "imageScore93", "imageScore94", "imageScore95", "imageScore96",
                  "imageScore97", "imageScore98", "imageScore99", "imageScore100",
                  )