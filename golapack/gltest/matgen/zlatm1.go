package matgen

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlatm1 computes the entries of D(1..N) as specified by
//    MODE, COND and IRSIGN. IDIST and ISEED determine the generation
//    of random numbers. Zlatm1 is called by ZLATMR to generate
//    random test matrices for LAPACK programs.
func Zlatm1(mode int, cond float64, irsign, idist int, iseed *[]int, d *mat.CVector, n int) (err error) {
	var ctemp complex128
	var alpha, one, temp float64
	var i int

	one = 1.0

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
	} else if (mode == 6 || mode == -6) && (idist < 1 || idist > 4) {
		err = fmt.Errorf("(mode == 6 || mode == -6) && (idist < 1 || idist > 4): mode=%v, idist=%v", mode, idist)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	}

	if err != nil {
		gltest.Xerbla2("Zlatm1", err)
		return
	}

	//     Compute D according to COND and MODE
	if mode != 0 {
		switch abs(mode) {
		case 1:
			goto label10
		case 2:
			goto label30
		case 3:
			goto label50
		case 4:
			goto label70
		case 5:
			goto label90
		case 6:
			goto label110
		}

		//        One large D value:
	label10:
		;
		for i = 1; i <= n; i++ {
			d.SetRe(i-1, one/cond)
		}
		d.SetRe(0, one)
		goto label120

		//        One small D value:
	label30:
		;
		for i = 1; i <= n; i++ {
			d.SetRe(i-1, one)
		}
		d.SetRe(n-1, one/cond)
		goto label120

		//        Exponentially distributed D values:
	label50:
		;
		d.SetRe(0, one)
		if n > 1 {
			alpha = math.Pow(cond, -one/float64(n-1))
			for i = 2; i <= n; i++ {
				d.SetRe(i-1, math.Pow(alpha, float64(i-1)))
			}
		}
		goto label120

		//        Arithmetically distributed D values:
	label70:
		;
		d.SetRe(0, one)
		if n > 1 {
			temp = one / cond
			alpha = (one - temp) / float64(n-1)
			for i = 2; i <= n; i++ {
				d.SetRe(i-1, float64(n-i)*alpha+temp)
			}
		}
		goto label120

		//        Randomly distributed D values on ( 1/COND , 1):
	label90:
		;
		alpha = math.Log(one / cond)
		for i = 1; i <= n; i++ {
			d.SetRe(i-1, math.Exp(alpha*Dlaran(iseed)))
		}
		goto label120

		//        Randomly distributed D values from IDIST
	label110:
		;
		golapack.Zlarnv(idist, iseed, n, d)

	label120:
		;

		//        If MODE neither -6 nor 0 nor 6, and IRSIGN = 1, assign
		//        random signs to D
		if (mode != -6 && mode != 0 && mode != 6) && irsign == 1 {
			for i = 1; i <= n; i++ {
				ctemp = Zlarnd(3, *iseed)
				d.Set(i-1, d.Get(i-1)*(ctemp/complex(cmplx.Abs(ctemp), 0)))
			}
		}

		//        Reverse if MODE < 0
		if mode < 0 {
			for i = 1; i <= n/2; i++ {
				ctemp = d.Get(i - 1)
				d.Set(i-1, d.Get(n+1-i-1))
				d.Set(n+1-i-1, ctemp)
			}
		}

	}

	return
}
