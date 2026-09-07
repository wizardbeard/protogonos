package scape

const FlatlandNeuralCost = 100

func FlatlandSpeak(avatar FlatlandAvatar, value float64) FlatlandAvatar {
	avatar.Sound = value
	return avatar
}

func FlatlandGestaltOutput(avatar FlatlandAvatar, gestalt []float64) FlatlandAvatar {
	avatar.Gestalt = append([]float64(nil), gestalt...)
	return avatar
}

func FlatlandCreateOffspring(avatar FlatlandAvatar, createValue float64) FlatlandAvatar {
	avatar.OffspringRequested = createValue > 0
	avatar.OffspringGranted = false
	avatar.OffspringCost = 0
	avatar.OffspringParentID = ""
	if createValue <= 0 {
		return avatar
	}

	offspringCost := float64(avatar.Stats * FlatlandNeuralCost)
	avatar.OffspringCost = offspringCost + 1000
	avatar.OffspringParentID = avatar.ID
	if avatar.Energy > offspringCost+1000 {
		avatar.Energy -= offspringCost + 1000
		avatar.OffspringGranted = true
		return avatar
	}
	avatar.Energy -= 50
	return avatar
}

func FlatlandSpear(avatar FlatlandAvatar, value float64) FlatlandAvatar {
	if value <= 0 {
		avatar.Spear = false
		return avatar
	}
	if avatar.Energy > 100 {
		avatar.Energy -= 10
		avatar.Spear = true
		return avatar
	}
	avatar.Energy--
	avatar.Spear = false
	return avatar
}

func FlatlandShoot(avatar FlatlandAvatar, value float64) FlatlandAvatar {
	if value <= 0 {
		return avatar
	}
	if avatar.Energy > 100 {
		avatar.Energy -= 20
		return avatar
	}
	avatar.Energy--
	return avatar
}
