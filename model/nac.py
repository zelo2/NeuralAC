# author@Zelo2
import torch
import torch.nn as nn
import torch.nn.functional as F



def one_to_all_teammate_index(team_size):
    index_target, index_teammate = [], []
    for i in range(team_size):
        for j in range(team_size):
            if i == j:
                continue
            index_target.append(i)
            index_teammate.append(j)
    return index_target, index_teammate


def one_to_all_enemy_index(team_size):
    index_target, index_teammate = [], []
    for i in range(team_size):
        for j in range(team_size):
            index_target.append(i)
            index_teammate.append(j+team_size)
    return index_target, index_teammate


class attention_net(nn.Module):  # weight = viT * W * vj
    def __init__(self, embed_dim, team_size, coop=True):  # embedding size = 20
        super(attention_net, self).__init__()
        assert (team_size > 1)  # 避免异常
        self.embed_dim = embed_dim
        self.team_size = team_size
        self.coop = coop

        '''如果是cooperation计算，里面的数据是team_size - 1为一组的，因为不包含自己和自己的合作效应。'''
        self.length1 = self.team_size
        self.length2 = self.team_size if not coop else self.team_size - 1

        self.attention_layer = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, A_team, B_team):  # [batch_size, (team_size - 1) * team_size, emd_dim]
        assert A_team.shape == B_team.shape
        A_team = A_team.view(-1, self.length1, self.length2, self.embed_dim)
        B_team = B_team.view(-1, self.length1, self.length2, self.embed_dim)
        weight = self.attention_layer(A_team) * B_team  # [batch_size, team_size, team_size-1, emd_dim]
        weight = weight.sum(dim=3)  # [batch_size, team_size, team_size-1]
        weight = F.softmax(weight, dim=2)  # [batch_size, team_size, TEAM_SIZE-1]
        result = weight.view(-1, self.length1 * self.length2)  # [batch_size, (team_size - 1) * team_size]

        return result





class coop_effect(nn.Module):
    def __init__(self, embed_dim, hero_num, team_size, attention=True):  # embedding size = 20

        super(coop_effect, self).__init__()
        assert (hero_num > 1 and team_size > 1)  # 避免异常
        self.embed_dim = embed_dim
        self.hero_num = hero_num
        self.team_size = team_size
        self.mlp_coop = nn.Sequential(nn.Linear(self.embed_dim, 50), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(50, 1), nn.ReLU())
        self.attention = attention
        self.index_target, self.index_teammate = one_to_all_teammate_index(self.team_size)
        #  target_id [0,0,0,0,1,1,1,1...] teammate_id [1,2,3,4,0,2,3,4...]

        '''Attention'''
        self.coop_attention_layer = attention_net(self.embed_dim, self.team_size, coop=True)

        '''Embedding'''
        self.cooperation_emd = nn.Embedding(self.hero_num, self.embed_dim)

    def forward(self, team_hero_id):  # [batch_size, team_size]
        target_emd = self.cooperation_emd(team_hero_id[:, self.index_target])
        teammate_emd = self.cooperation_emd(team_hero_id[:, self.index_teammate])  # [batch_size, (team_size-1) * team_size, emd_dim]
        cooperation_effect = self.mlp_coop(target_emd * teammate_emd).squeeze()  # [batch_size, (team_size-1) * team_size]

        if self.attention:
            cooperation_effect *= self.coop_attention_layer(target_emd, teammate_emd)
            pass  # To be continue


        result = torch.sum(cooperation_effect, dim=1, keepdim=True)  # sum each row [batch_size, 1]

        return result

class comp_effect(nn.Module):
    def __init__(self, embed_dim, hero_num, team_size, attention=True):  # embedding size = 20
        super(comp_effect, self).__init__()
        assert (hero_num > 1 and team_size > 1)  # 避免异常
        self.embed_dim = embed_dim
        self.hero_num = hero_num
        self.team_size = team_size
        self.mlp_comp = nn.Sequential(nn.Linear(self.embed_dim, 50), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(50, 1), nn.ReLU())
        self.attention = attention
        self.index_target, self.index_enemy = one_to_all_enemy_index(self.team_size)
        #  target_id [0,0,0,0,0,1,1,1,1,1...] teammate_id [5,6,7,8,9,5,6,7,8,9...]

        '''Attention'''
        self.comp_attention_layer = attention_net(self.embed_dim, self.team_size, coop=False)  # competition

        '''Embedding Layer'''
        self.strength_emd = nn.Embedding(self.hero_num, self.embed_dim)
        self.weakness_emd = nn.Embedding(self.hero_num, self.embed_dim)

    def forward(self, two_team_hero_id):
        A_team_hero_id = two_team_hero_id[:, self.index_target]
        B_team_hero_id = two_team_hero_id[:, self.index_enemy]

        A_team_ST_emd = self.strength_emd(A_team_hero_id)  # [batch_size, team_size * 5, emd_dim]
        A_team_WK_emd = self.weakness_emd(A_team_hero_id)

        B_team_ST_emd = self.strength_emd(B_team_hero_id)
        B_team_WK_emd = self.weakness_emd(B_team_hero_id)

        competition_effect = self.mlp_comp(A_team_ST_emd * B_team_WK_emd).squeeze()  # [batch_size, team_size * team_size]

        if self.attention:
            competition_effect *= self.comp_attention_layer(A_team_ST_emd, B_team_WK_emd)

        result = torch.sum(competition_effect, dim=1, keepdim=True)  # [batch_size, 1]

        return result



class nac_net(nn.Module):
    def __init__(self, embed_dim, hero_num, team_size, attention=True):  # embedding size = 20
        super(nac_net, self).__init__()
        self.embed_dim = embed_dim
        self.hero_num = hero_num
        self.team_size = team_size
        self.attention = attention

        self.cooperation_net = coop_effect(self.embed_dim, self.hero_num, self.team_size, attention=self.attention)
        self.competition_net = comp_effect(self.embed_dim, self.hero_num, self.team_size, attention=self.attention)

        '''Embedding stage'''
        self.ability_emd = nn.Embedding(self.hero_num, 1)


    def forward(self, two_team_hero_id):
        A_team_hero_id = two_team_hero_id[:, :self.team_size]
        B_team_hero_id = two_team_hero_id[:, self.team_size:]

        A_team_ability = self.ability_emd(A_team_hero_id).view(len(two_team_hero_id), -1)  # [batch_size, team_size]
        B_team_ability = self.ability_emd(B_team_hero_id).view(len(two_team_hero_id), -1)  # [batch_size, team_size]

        A_team_ability = A_team_ability.sum(dim=1, keepdim=True)  # [batch_size, 1]
        B_team_ability = B_team_ability.sum(dim=1, keepdim=True)  # [batch_size, 1]

        A_coop_effect = self.cooperation_net(A_team_hero_id)
        A_comp_effect = self.competition_net(two_team_hero_id)

        B_coop_effect = self.cooperation_net(B_team_hero_id)

        B_comp_effect = self.competition_net(torch.cat((B_team_hero_id, A_team_hero_id), dim=1))  # equal torch.flip()

        Sa = A_team_ability + A_coop_effect + A_comp_effect
        Sb = B_team_ability + B_coop_effect + B_comp_effect


        results = torch.sigmoid(Sa - Sb).view(-1)

        return results  # batch_size






