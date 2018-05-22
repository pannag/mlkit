package com.google.mlkitcustomdemo.mlkitcustom;

import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseModelOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;
import com.google.firebase.ml.custom.model.FirebaseLocalModelSource;
import java.util.Arrays;
import java.util.Random;

/**
 * A simple app that runs a tflite model.
 *
 * <p>The tflite model takes a 64-dim input and returns the sum of all the inputs. This is like a
 * logistic regression model without the softmax layer and where all the weights are 1.0 and bias is
 * 0.
 *
 * <p>To make the test easy, you can enter a number that will be replicated 64 times and passed as
 * input to the model. The expected output is 64 * input number. If no input is given, then 64
 * random numbers are used as input to the model. </p>
 */
public class MainActivity extends AppCompatActivity {

  private static final String TAG = "MainActivity";
  /** Name of the model file stored in Assets. */
  private static final String MODEL_PATH = "sum_up_model.tflite"; // "logreg.tflite";
  private static final String LOCAL_MODEL_NAME = "local_model";

  // Define some constants for input / output dimensions of the model.
  private static final int DIM_BATCH_SIZE = 1;
  private static final int DIM_INPUT_SIZE = 64;
  private static final int DIM_OUTPUT_SIZE = 1;

  private Button mRun;
  private TextView mTextView;
  private FirebaseModelInputOutputOptions mInputOutputOptions;
  private FirebaseModelInterpreter mInterpreter;
  private EditText mEditText;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    mTextView = findViewById(R.id.textview);
    mEditText = findViewById(R.id.editText);

    mRun = findViewById(R.id.button);
    mRun.setOnClickListener(new OnClickListener() {
      @Override
      public void onClick(View v) {
        runModelInference();
      }
    });

    FirebaseLocalModelSource localSource = new FirebaseLocalModelSource.Builder(LOCAL_MODEL_NAME)
        .setAssetFilePath(MODEL_PATH)  // Or setFilePath if you downloaded from your host
        .build();
    FirebaseModelManager.getInstance().registerLocalModelSource(localSource);

    // Build the model interpreter
    FirebaseModelOptions modelOptions =
        new FirebaseModelOptions.Builder()
            .setLocalModelName(LOCAL_MODEL_NAME)
            .build();

    try {
      mInputOutputOptions =
          new FirebaseModelInputOutputOptions.Builder()
              .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{DIM_BATCH_SIZE, DIM_INPUT_SIZE})
              .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{DIM_BATCH_SIZE, DIM_OUTPUT_SIZE})
              .build();

      mInterpreter =
          FirebaseModelInterpreter.getInstance(modelOptions);
      mTextView.setText("Model Interpreter Created..");

    } catch (FirebaseMLException e) {
      showToast("Error while setting up the model interpreter!");
      e.printStackTrace();
    }

  }

  private void runModelInference() {
    mTextView.setText("Running Model Inference..");
    if (mInterpreter == null) {
      Log.e(TAG, "Model Interpreter has not been initialized.");
      return;
    }

    float[] inputVec = new float[DIM_INPUT_SIZE];
    Random random = new Random();
    final String str = mEditText.getText().toString();
    if (str.matches("")) {
      Toast.makeText(this, "You did not enter an input number! Using random numbers.",
          Toast.LENGTH_SHORT).show();
    }
    for (int i = 0; i < inputVec.length; ++i) {
      if (str.matches("")) {
        inputVec[i] = random.nextFloat();
      } else {
        inputVec[i] = Float.valueOf(str);
      }
    }

    float[][] inputData = new float[DIM_BATCH_SIZE][DIM_INPUT_SIZE];
    inputData[0] = inputVec;

    try {
      FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(inputData).build();
      mInterpreter
          .run(inputs, mInputOutputOptions)
          .addOnSuccessListener(new OnSuccessListener<FirebaseModelOutputs>() {
            @Override
            public void onSuccess(FirebaseModelOutputs result) {
              float[][] output = result.getOutput(0);
              float[] logits = output[0];
              Log.i(TAG, "Sum = " + logits[0]);
              Log.i(TAG, Arrays.toString(logits));
              mTextView.setText("Sum = " + logits[0]);
            }
          })
          .addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
              mTextView.setText("Inference Failed!");
              e.printStackTrace();
            }
          });
    } catch (FirebaseMLException e) {
      showToast("Something went wrong during inference!");
      mTextView.setText("Done");
      e.printStackTrace();
    }

  }

  private void showToast(String message) {
    Toast.makeText(getApplicationContext(), message, Toast.LENGTH_SHORT).show();
  }


}
