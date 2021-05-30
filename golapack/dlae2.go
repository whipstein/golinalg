package golapack

import "math"

// Dlae2 computes the eigenvalues of a 2-by-2 symmetric matrix
//    [  A   B  ]
//    [  B   C  ].
// On return, RT1 is the eigenvalue of larger absolute value, and RT2
// is the eigenvalue of smaller absolute value.
func Dlae2(a, b, c, rt1, rt2 *float64) {
	var ab, acmn, acmx, adf, df, half, one, rt, sm, tb, two, zero float64

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

		//        Order of execution important.
		//        To get fully accurate smaller eigenvalue,
		//        next line needs to be executed in higher precision.
		(*rt2) = (acmx/(*rt1))*acmn - ((*b)/(*rt1))*(*b)
	} else if sm > zero {
		(*rt1) = half * (sm + rt)

		//        Order of execution important.
		//        To get fully accurate smaller eigenvalue,
		//        next line needs to be executed in higher precision.
		(*rt2) = (acmx/(*rt1))*acmn - ((*b)/(*rt1))*(*b)
	} else {
		//        Includes case RT1 = RT2 = 0
		(*rt1) = half * rt
		(*rt2) = -half * rt
	}
}
