package golapack

import "math"

// Dlaev2 computes the eigendecomposition of a 2-by-2 symmetric matrix
//    [  A   B  ]
//    [  B   C  ].
// On return, RT1 is the eigenvalue of larger absolute value, RT2 is the
// eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right
// eigenvector for RT1, giving the decomposition
//
//    [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
//    [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ].
func Dlaev2(a, b, c, rt1, rt2, cs1, sn1 *float64) {
	var ab, acmn, acmx, acs, adf, cs, ct, df, half, one, rt, sm, tb, tn, two, zero float64
	var sgn1, sgn2 int

	one = 1.0
	two = 2.0
	zero = 0.0
	half = 0.5

	//     Compute the eigenvalues
	sm = (*a) + (*c)
	df = (*a) - (*c)
	adf = math.Abs(df)
	tb = (*b) + (*b)
	ab = math.Abs(tb)
	if math.Abs(*a) > math.Abs(*c) {
		acmx = (*a)
		acmn = (*c)
	} else {
		acmx = (*c)
		acmn = (*a)
	}
	if adf > ab {
		rt = adf * math.Sqrt(one+math.Pow(ab/adf, 2))
	} else if adf < ab {
		rt = ab * math.Sqrt(one+math.Pow(adf/ab, 2))
	} else {
		//        Includes case AB=ADF=0
		rt = ab * math.Sqrt(two)
	}
	if sm < zero {
		(*rt1) = half * (sm - rt)
		sgn1 = -1

		//        Order of execution important.
		//        To get fully accurate smaller eigenvalue,
		//        next line needs to be executed in higher precision.
		(*rt2) = (acmx/(*rt1))*acmn - ((*b)/(*rt1))*(*b)
	} else if sm > zero {
		(*rt1) = half * (sm + rt)
		sgn1 = 1

		//        Order of execution important.
		//        To get fully accurate smaller eigenvalue,
		//        next line needs to be executed in higher precision.
		(*rt2) = (acmx/(*rt1))*acmn - ((*b)/(*rt1))*(*b)
	} else {
		//        Includes case RT1 = RT2 = 0
		(*rt1) = half * rt
		(*rt2) = -half * rt
		sgn1 = 1
	}

	//     Compute the eigenvector
	if df >= zero {
		cs = df + rt
		sgn2 = 1
	} else {
		cs = df - rt
		sgn2 = -1
	}
	acs = math.Abs(cs)
	if acs > ab {
		ct = -tb / cs
		(*sn1) = one / math.Sqrt(one+ct*ct)
		(*cs1) = ct * (*sn1)
	} else {
		if ab == zero {
			(*cs1) = one
			(*sn1) = zero
		} else {
			tn = -cs / tb
			(*cs1) = one / math.Sqrt(one+tn*tn)
			(*sn1) = tn * (*cs1)
		}
	}
	if sgn1 == sgn2 {
		tn = (*cs1)
		(*cs1) = -(*sn1)
		(*sn1) = tn
	}
}
