package matgen

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlatm7 computes the entries of D as specified by MODE
//    COND and IRSIGN. IDIST and ISEED determine the generation
//    of random numbers. DLATM7 is called by DLATMT to generate
//    random test matrices.
func Dlatm7(mode *int, cond *float64, irsign, idist *int, iseed *[]int, d *mat.Vector, n, rank, info *int) {
	var alpha, half, one, temp, zero float64
	var i int

	one = 1.0
	zero = 0.0
	half = 0.5

	//     Decode and Test the input parameters. Initialize flags & seed.
	(*info) = 0

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Set INFO if an error
	if (*mode) < -6 || (*mode) > 6 {
		(*info) = -1
	} else if ((*mode) != -6 && (*mode) != 0 && (*mode) != 6) && ((*irsign) != 0 && (*irsign) != 1) {
		(*info) = -2
	} else if ((*mode) != -6 && (*mode) != 0 && (*mode) != 6) && (*cond) < one {
		(*info) = -3
	} else if ((*mode) == 6 || (*mode) == -6) && ((*idist) < 1 || (*idist) > 3) {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -7
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DLATM7"), -(*info))
		return
	}

	//     Compute D according to COND and MODE
	if (*mode) != 0 {
		switch absint(*mode) {
		case 1:
			goto label100
		case 2:
			goto label130
		case 3:
			goto label160
		case 4:
			goto label190
		case 5:
			goto label210
		case 6:
			goto label230
		}

		//        One large D value:
	label100:
		;
		for i = 2; i <= (*rank); i++ {
			d.Set(i-1, one/(*cond))
		}
		for i = (*rank) + 1; i <= (*n); i++ {
			d.Set(i-1, zero)
		}
		d.Set(0, one)
		goto label240

		//        One small D value:
	label130:
		;
		for i = 1; i <= (*rank)-1; i++ {
			d.Set(i-1, one)
		}
		for i = (*rank) + 1; i <= (*n); i++ {
			d.Set(i-1, zero)
		}
		d.Set((*rank)-1, one/(*cond))
		goto label240

		//        Exponentially distributed D values:
	label160:
		;
		d.Set(0, one)
		if (*n) > 1 && (*rank) > 1 {
			alpha = math.Pow(*cond, -one/float64((*rank)-1))
			for i = 2; i <= (*rank); i++ {
				d.Set(i-1, math.Pow(alpha, float64(i-1)))
			}
			for i = (*rank) + 1; i <= (*n); i++ {
				d.Set(i-1, zero)
			}
		}
		goto label240

		//        Arithmetically distributed D values:
	label190:
		;
		d.Set(0, one)
		if (*n) > 1 {
			temp = one / (*cond)
			alpha = (one - temp) / float64((*n)-1)
			for i = 2; i <= (*n); i++ {
				d.Set(i-1, float64((*n)-i)*alpha+temp)
			}
		}
		goto label240

		//        Randomly distributed D values on ( 1/COND , 1):
	label210:
		;
		alpha = math.Log(one / (*cond))
		for i = 1; i <= (*n); i++ {
			d.Set(i-1, math.Exp(alpha*Dlaran(iseed)))
		}
		goto label240

		//        Randomly distributed D values from IDIST
	label230:
		;
		golapack.Dlarnv(idist, iseed, n, d)

	label240:
		;

		//        If MODE neither -6 nor 0 nor 6, and IRSIGN = 1, assign
		//        random signs to D
		if ((*mode) != -6 && (*mode) != 0 && (*mode) != 6) && (*irsign) == 1 {
			for i = 1; i <= (*n); i++ {
				temp = Dlaran(iseed)
				if temp > half {
					d.Set(i-1, -d.Get(i-1))
				}
			}
		}

		//        Reverse if MODE < 0
		if (*mode) < 0 {
			for i = 1; i <= (*n)/2; i++ {
				temp = d.Get(i - 1)
				d.Set(i-1, d.Get((*n)+1-i-1))
				d.Set((*n)+1-i-1, temp)
			}
		}

	}
}
