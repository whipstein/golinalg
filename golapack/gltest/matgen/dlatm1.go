package matgen

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlatm1 computes the entries of D(1..N) as specified by
//    MODE, COND and IRSIGN. IDIST and ISEED determine the generation
//    of random numbers. Dlatm1 is called by DLATMR to generate
//    random test matrices for LAPACK programs.
func Dlatm1(mode int, cond float64, irsign, idist int, iseed *[]int, d *mat.Vector, n int) (err error) {
	var alpha, half, one, temp float64
	var i int

	one = 1.0
	half = 0.5

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Set INFO if an error
	if mode < -6 || mode > 6 {
		err = fmt.Errorf("mode < -6 || mode > 6: mode=%v", mode)
	} else if (mode != -6 && mode != 0 && mode != 6) && (irsign != 0 && irsign != 1) {
		err = fmt.Errorf("(mode != -6 && mode != 0 && mode != 6) && (irsign != 0 && irsign != 1): mode=%v, irsign=%v", mode, irsign)
	} else if (mode != -6 && mode != 0 && mode != 6) && cond < one {
		err = fmt.Errorf("(mode != -6 && mode != 0 && mode != 6) && cond < one: mode=%v, cond=%v", mode, cond)
	} else if (mode == 6 || mode == -6) && (idist < 1 || idist > 3) {
		err = fmt.Errorf("(mode == 6 || mode == -6) && (idist < 1 || idist > 3): mode=%v, idist=%v", mode, idist)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	}

	if err != nil {
		gltest.Xerbla2("Dlatm1", err)
		return
	}

	//     Compute D according to COND and MODE
	if mode != 0 {
		switch abs(mode) {
		case 1:
			//        One large D value:
			for i = 1; i <= n; i++ {
				d.Set(i-1, one/cond)
			}
			d.Set(0, one)
		case 2:
			//        One small D value:
			for i = 1; i <= n; i++ {
				d.Set(i-1, one)
			}
			d.Set(n-1, one/cond)
		case 3:
			//        Exponentially distributed D values:
			d.Set(0, one)
			if n > 1 {
				alpha = math.Pow(cond, -one/float64(n-1))
				for i = 2; i <= n; i++ {
					d.Set(i-1, math.Pow(alpha, float64(i-1)))
				}
			}
		case 4:
			//        Arithmetically distributed D values:
			d.Set(0, one)
			if n > 1 {
				temp = one / cond
				alpha = (one - temp) / float64(n-1)
				for i = 2; i <= n; i++ {
					d.Set(i-1, float64(n-i)*alpha+temp)
				}
			}
		case 5:
			//        Randomly distributed D values on ( 1/COND , 1):
			alpha = math.Log(one / cond)
			for i = 1; i <= n; i++ {
				d.Set(i-1, math.Exp(alpha*Dlaran(iseed)))
			}
		case 6:
			//        Randomly distributed D values from IDIST
			golapack.Dlarnv(idist, iseed, n, d)
		}

		//        If MODE neither -6 nor 0 nor 6, and IRSIGN = 1, assign
		//        random signs to D
		if (mode != -6 && mode != 0 && mode != 6) && irsign == 1 {
			for i = 1; i <= n; i++ {
				temp = Dlaran(iseed)
				if temp > half {
					d.Set(i-1, -d.Get(i-1))
				}
			}
		}

		//        Reverse if MODE < 0
		if mode < 0 {
			for i = 1; i <= n/2; i++ {
				temp = d.Get(i - 1)
				d.Set(i-1, d.Get(n+1-i-1))
				d.Set(n+1-i-1, temp)
			}
		}

	}

	return
}
